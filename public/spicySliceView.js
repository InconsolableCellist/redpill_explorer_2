export class SpicySliceView {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.selectedSlice = null;
        this.hoveredNode = null;
        this.sliceCanvas = null;
        this.sliceCanvasMousePos = { x: 0, y: 0 };
        this.tooltip = document.getElementById('tooltip');
        this.thumbnailViewer = document.getElementById('thumbnailViewer');
        this.thumbnailImage = document.getElementById('thumbnailImage');
        this.setupSliceViewer();
    }

    setupSliceViewer() {
        const viewer = document.getElementById('sliceViewer');
        const canvas = document.getElementById('sliceCanvas');
        canvas.width = 380;
        canvas.height = 380;
        this.sliceCanvas = canvas;

        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.textContent = '×';
        closeBtn.className = 'slice-close-btn';
        closeBtn.onclick = () => this.closeSliceView();
        viewer.insertBefore(closeBtn, viewer.firstChild);

        // Add slice view event listeners
        canvas.addEventListener('mouseleave', () => {
            this.hideTooltip();
            this.hideThumbnail();
        });
        canvas.addEventListener('mousemove', (event) => this.handleSliceCanvasMouseMove(event));
        canvas.addEventListener('click', (event) => this.handleSliceCanvasClick(event));
        canvas.addEventListener('dblclick', (event) => this.handleSliceCanvasDoubleClick(event));
    }

    closeSliceView() {
        const viewer = document.getElementById('sliceViewer');
        viewer.style.display = 'none';
        this.hideThumbnail();
        if (this.selectedSlice) {
            this.selectedSlice.material.opacity = 0.1;
            this.selectedSlice = null;
        }
    }

    showSliceView(sliceMesh) {
        const viewer = document.getElementById('sliceViewer');
        const title = document.getElementById('sliceTitle');
        const canvas = this.sliceCanvas;
        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw slice boundary
        ctx.beginPath();
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.arc(canvas.width/2, canvas.height/2, canvas.width/2 - 10, 0, Math.PI * 2);
        ctx.stroke();

        title.textContent = `Slice View - Spicy: ${sliceMesh.userData.level.toFixed(1)}`;

        // Draw nodes with tag-based visibility
        const scale = (canvas.width - 20) / (this.visualizer.SLICE_RADIUS * 2);
        const center = canvas.width / 2;

        sliceMesh.userData.items.forEach(item => {
            const node = this.visualizer.nodes.get(item.hash);
            if (!node) return;

            // Check if we should show this node based on selected tags
            let shouldShow = true;
            if (this.visualizer.selectedTags.size > 0) {
                const itemData = this.visualizer.nodeData.get(item.hash);
                const nodeTags = new Set(Object.keys(itemData.tags));
                // Check if node has ALL selected tags (AND operation)
                shouldShow = Array.from(this.visualizer.selectedTags)
                    .every(tag => nodeTags.has(tag));
            }

            if (shouldShow) {
                const x = center + node.userData.radius * Math.cos(node.userData.angle) * scale;
                const y = center + node.userData.radius * Math.sin(node.userData.angle) * scale;

                // Draw glow effect
                ctx.beginPath();
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
                const color = node.material.color;
                const opacity = node.material.opacity;
                const colorStr = `rgba(${color.r * 255}, ${color.g * 255}, ${color.b * 255}, ${opacity})`;
                gradient.addColorStop(0, colorStr);
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                ctx.fillStyle = gradient;
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();

                // Draw node center
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fillStyle = colorStr;
                ctx.fill();
            }
        });

        this.selectedSlice = sliceMesh;
        viewer.style.display = 'block';
    }

    getNodeAtCanvasPosition(x, y) {
        if (!this.selectedSlice) return null;

        const canvas = this.sliceCanvas;
        const rect = canvas.getBoundingClientRect();
        const scale = (canvas.width - 20) / (this.visualizer.SLICE_RADIUS * 2);
        const center = canvas.width / 2;

        // Convert to canvas coordinates
        const canvasX = x - rect.left;
        const canvasY = y - rect.top;

        // Convert to polar coordinates relative to center
        const relX = (canvasX - center) / scale;
        const relY = (canvasY - center) / scale;
        const radius = Math.sqrt(relX * relX + relY * relY);

        // Check if within slice radius
        if (radius > this.visualizer.SLICE_RADIUS) return null;

        // Find closest node
        let closestNode = null;
        let closestDistance = 10; // Threshold for node selection in pixels

        this.selectedSlice.userData.items.forEach(item => {
            const node = this.visualizer.nodes.get(item.hash);
            if (!node) return;

            const nodeX = center + node.userData.radius * Math.cos(node.userData.angle) * scale;
            const nodeY = center + node.userData.radius * Math.sin(node.userData.angle) * scale;

            const distance = Math.sqrt(
                Math.pow(canvasX - nodeX, 2) +
                Math.pow(canvasY - nodeY, 2)
            );

            if (distance < closestDistance) {
                closestDistance = distance;
                closestNode = node;
            }
        });

        return closestNode;
    }

    handleSliceCanvasMouseMove(event) {
        const node = this.getNodeAtCanvasPosition(event.clientX, event.clientY);
        if (node) {
            // Update visual feedback for hover
            if (this.hoveredNode !== node) {
                // Reset previous hover state
                if (this.hoveredNode) {
                    this.hoveredNode.scale.set(1, 1, 1);
                }
                // Set new hover state
                this.hoveredNode = node;
                node.scale.set(1.5, 1.5, 1.5);
            }
            this.updateTooltip(event, node);
            this.updateThumbnail(node);
        } else {
            // Reset hover state when not over any node
            if (this.hoveredNode) {
                this.hoveredNode.scale.set(1, 1, 1);
                this.hoveredNode = null;
            }
            this.hideTooltip();
            this.hideThumbnail();
        }

        // Store mouse position for reference
        this.sliceCanvasMousePos = {
            x: event.clientX,
            y: event.clientY
        };
    }

    handleSliceCanvasClick(event) {
        const node = this.getNodeAtCanvasPosition(event.clientX, event.clientY);
        if (node) {
            window.open(`/node-info/${node.userData.hash}`, '_blank');
        }
    }

    handleSliceCanvasDoubleClick(event) {
        const node = this.getNodeAtCanvasPosition(event.clientX, event.clientY);
        if (node) {
            window.open(`/node-info/${node.userData.hash}`, '_blank');
        }
    }

    updateTooltip(event, node) {
        if (node && node.userData.hash) {
            const itemData = this.visualizer.nodeData.get(node.userData.hash);
            if (itemData) {
                const description = itemData.description.length > 500
                    ? itemData.description.substring(0, 500) + '...'
                    : itemData.description;

                this.tooltip.innerHTML = `
                    <div><strong>Spicy Level:</strong> ${itemData.spicy.toFixed(2)}</div>
                    <div><strong>Description:</strong> ${description}</div>
                    <div><strong>Tags:</strong> ${Object.keys(itemData.tags).join(', ')}</div>
                `;

                // Position tooltip near mouse but ensure it stays within viewport
                const tooltipWidth = this.tooltip.offsetWidth;
                const tooltipHeight = this.tooltip.offsetHeight;
                const padding = 15;

                let left = event.clientX + padding;
                let top = event.clientY + padding;

                if (left + tooltipWidth > window.innerWidth) {
                    left = event.clientX - tooltipWidth - padding;
                }
                if (top + tooltipHeight > window.innerHeight) {
                    top = event.clientY - tooltipHeight - padding;
                }

                this.tooltip.style.left = `${left}px`;
                this.tooltip.style.top = `${top}px`;
                this.tooltip.style.display = 'block';
            }
        } else {
            this.hideTooltip();
        }
    }

    updateThumbnail(node) {
        if (node && node.userData.hash) {
            const hash = node.userData.hash;
            const thumbnailUrl = `/thumbnails/${hash[0]}/${hash[1]}/${hash[2]}/${hash.substring(3)}.jpg`;

            // Only update src if it's different to avoid flickering
            if (this.thumbnailImage.src !== thumbnailUrl) {
                this.thumbnailImage.src = thumbnailUrl;
            }

            this.thumbnailViewer.style.display = 'block';
        }
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    hideThumbnail() {
        this.thumbnailViewer.style.display = 'none';
    }
}