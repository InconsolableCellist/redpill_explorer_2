import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class SpicyInteractions {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.controls = null;
        this.tooltip = document.getElementById('tooltip');
        this.isTooltipVisible = false;
        this.setupOrbitControls();
        this.setupEventListeners();
        this.setupTagPanel();
        this.selectedSlice = null;
        this.hoveredSlice = null;
        this.setupSliceViewer();
        this.hoveredNode = null;
        this.sliceCanvasMousePos = { x: 0, y: 0 };

        // Camera parameters
        this.cameraRadius = 600;  // Distance from center axis
        this.cameraAngle = Math.PI / 4;  // 45 degrees from vertical
        this.cameraHeight = 500;  // Initial height
        this.rotationAngle = 0;   // Current rotation around vertical axis
    }

    setupOrbitControls() {
        // Initialize OrbitControls
        this.controls = new OrbitControls(
            this.visualizer.camera,
            this.visualizer.renderer.domElement
        );

        // Disable all built-in OrbitControls
        this.controls.enabled = false;

        // Set initial camera position
        this.updateCameraPosition();

        let isMouseDown = false;
        let previousX = 0;
        let previousY = 0;

        this.visualizer.renderer.domElement.addEventListener('mousedown', (event) => {
            if (event.button === 0) { // Left button only
                isMouseDown = true;
                previousX = event.clientX;
                previousY = event.clientY;
            }
        });

        this.visualizer.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;

            // Calculate deltas
            const deltaX = event.clientX - previousX;
            const deltaY = event.clientY - previousY;

            // Update rotation around vertical axis
            this.rotationAngle -= deltaX * 0.01;

            // Update height
            this.cameraHeight -= deltaY * 2;

            // Update camera position using new values
            this.updateCameraPosition();

            previousX = event.clientX;
            previousY = event.clientY;
        });

        document.addEventListener('mouseup', (event) => {
            if (event.button === 0) {
                isMouseDown = false;
            }
        });

        // Handle zoom with mouse wheel
        this.visualizer.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();
            const zoomSpeed = 0.1;
            const delta = event.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;
            this.cameraRadius = Math.max(200, Math.min(1000, this.cameraRadius * delta));
            this.updateCameraPosition();
        });

        // Prevent context menu
        this.visualizer.renderer.domElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });
    }

    updateCameraPosition() {
        // Calculate camera position based on spherical coordinates
        const horizontalRadius = this.cameraRadius * Math.sin(this.cameraAngle);
        const x = horizontalRadius * Math.cos(this.rotationAngle);
        const z = horizontalRadius * Math.sin(this.rotationAngle);
        const y = this.cameraHeight + this.cameraRadius * Math.cos(this.cameraAngle);

        // Update camera position
        this.visualizer.camera.position.set(x, y, z);

        // Update look-at target (always look at center axis at current height)
        this.visualizer.camera.lookAt(0, this.cameraHeight, 0);
        this.controls.target.set(0, this.cameraHeight, 0);
    }

    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.visualizer.onWindowResize();
        }, false);

        this.visualizer.renderer.domElement.addEventListener('mousemove', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);

            if (intersectedObject) {
                if (intersectedObject.userData.type === 'node') {
                    this.visualizer.highlightNode(intersectedObject);
                    this.updateTooltip(event, intersectedObject);
                } else {
                    this.visualizer.unhighlightNode();
                    this.hideTooltip();
                }
            } else {
                this.visualizer.unhighlightNode();
                this.hideTooltip();
            }
        });

        this.visualizer.renderer.domElement.addEventListener('mouseleave', () => {
            this.visualizer.unhighlightNode();
            this.hideTooltip();
        });

        // Main view click handling
        this.visualizer.renderer.domElement.addEventListener('click', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (!intersectedObject) return;

            if (intersectedObject.userData.type === 'slice') {
                if (this.selectedSlice && this.selectedSlice !== intersectedObject) {
                    this.selectedSlice.material.opacity = 0.1;
                }
                this.selectedSlice = intersectedObject;
                intersectedObject.material.opacity = 0.4;
                this.showSliceView(intersectedObject);
            } else if (intersectedObject.userData.type === 'node') {
                this.handleNodeClick(intersectedObject);
            }
        });

        // Main view double click
        this.visualizer.renderer.domElement.addEventListener('dblclick', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (intersectedObject && intersectedObject.userData.type === 'node') {
                this.handleNodeDoubleClick(intersectedObject);
            }
        });

        // Keyboard events
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.closeSliceView();
                this.visualizer.selectedTags.clear();
                this.updateSelectedTagsPanel();
                this.visualizer.updateSelection();
            }
        });
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
        canvas.addEventListener('mouseleave', () => this.hideTooltip());
        canvas.addEventListener('mousemove', (event) => this.handleSliceCanvasMouseMove(event));
        canvas.addEventListener('click', (event) => this.handleSliceCanvasClick(event));
        canvas.addEventListener('dblclick', (event) => this.handleSliceCanvasDoubleClick(event));
    }

    closeSliceView() {
        const viewer = document.getElementById('sliceViewer');
        viewer.style.display = 'none';
        if (this.selectedSlice) {
            this.selectedSlice.material.opacity = 0.1;
            this.selectedSlice = null;
        }
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

    handleSliceCanvasDoubleClick(event) {
        const node = this.getNodeAtCanvasPosition(event.clientX, event.clientY);
        if (node) {
            this.handleNodeDoubleClick(node);
        }
    }

    handleNodeDoubleClick(node) {
        window.open(`/node-info/${node.userData.hash}`, '_blank');
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
                const nodeTags = Object.keys(itemData.tags);
                shouldShow = Array.from(this.visualizer.selectedTags)
                    .some(tag => nodeTags.includes(tag));
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

        viewer.style.display = 'block';
    }

    updateTooltip(event, node) {
        if (node && node.userData.hash) {
            const itemData = this.visualizer.nodeData.get(node.userData.hash);
            if (itemData) {
                // Create tooltip content
                const description = itemData.description.length > 200
                    ? itemData.description.substring(0, 200) + '...'
                    : itemData.description;

                this.tooltip.innerHTML = `
                <div><strong>Spicy Level:</strong> ${itemData.spicy.toFixed(2)}</div>
                <div><strong>Tags:</strong> ${Object.keys(itemData.tags).join(', ')}</div>
                <div><strong>Description:</strong> ${description}</div>
            `;

                // Position tooltip near mouse but ensure it stays within viewport
                const tooltipWidth = this.tooltip.offsetWidth;
                const tooltipHeight = this.tooltip.offsetHeight;
                const padding = 15; // Space between mouse and tooltip

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
                this.isTooltipVisible = true;
            }
        } else {
            this.hideTooltip();
        }
    }

    hideTooltip() {
        if (this.isTooltipVisible) {
            this.tooltip.style.display = 'none';
            this.isTooltipVisible = false;
        }
    }

    async setupTagPanel() {
        // Load tags data
        const response = await fetch('/tags');
        const tagsData = await response.json();

        const tagList = document.getElementById('tagList');
        const tagSearch = document.getElementById('tagSearch');
        const selectedTagsPanel = document.getElementById('selectedTags');

        // Sort tags by size (number of items)
        const sortedTags = Object.entries(tagsData)
            .sort(([, a], [, b]) => b.itemIds.length - a.itemIds.length)
            .map(([tag]) => tag);

        // Create tag elements
        sortedTags.forEach(tag => {
            const tagElement = document.createElement('div');
            tagElement.className = 'tag-item';
            const count = tagsData[tag].itemIds.length;
            tagElement.innerHTML = `${tag} <span class="tag-count">(${count})</span>`;

            tagElement.addEventListener('click', () => {
                if (!this.visualizer.selectedTags.has(tag)) {
                    this.visualizer.selectedTags.add(tag);
                    this.updateSelectedTagsPanel();
                    this.visualizer.updateSelection();
                    tagElement.classList.add('selected');

                    // Update slice view if it's open
                    if (this.selectedSlice) {
                        this.showSliceView(this.selectedSlice);
                    }
                }
            });

            tagList.appendChild(tagElement);
        });

        // Search functionality
        tagSearch.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            Array.from(tagList.children).forEach(tagElement => {
                const tagText = tagElement.textContent.toLowerCase();
                const matches = tagText.includes(searchTerm);
                tagElement.style.display = matches ? 'block' : 'none';
            });
        });

        // Clear search on escape
        tagSearch.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                tagSearch.value = '';
                Array.from(tagList.children).forEach(tagElement => {
                    tagElement.style.display = 'block';
                });
            }
        });
    }

    updateSelectedTagsPanel() {
        const panel = document.getElementById('selectedTags');
        panel.innerHTML = '';

        this.visualizer.selectedTags.forEach(tag => {
            const tagElement = document.createElement('div');
            tagElement.className = 'selected-tag';
            tagElement.innerHTML = `${tag} <span class="remove-tag">×</span>`;

            tagElement.addEventListener('click', () => {
                this.visualizer.selectedTags.delete(tag);
                this.updateSelectedTagsPanel();
                this.visualizer.updateSelection();

                // Update tag list selection state
                const tagListItem = Array.from(document.getElementsByClassName('tag-item'))
                    .find(el => el.textContent.includes(tag));
                if (tagListItem) {
                    tagListItem.classList.remove('selected');
                }

                // Update slice view if it's open
                if (this.selectedSlice) {
                    this.showSliceView(this.selectedSlice);
                }
            });

            panel.appendChild(tagElement);
        });
    }

    handleNodeClick(node) {
        // Open node details instead of adding tag
        window.open(`/node-info/${node.userData.hash}`, '_blank');
    }
    // Add these methods to the SpicyInteractions class

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
        } else {
            // Reset hover state when not over any node
            if (this.hoveredNode) {
                this.hoveredNode.scale.set(1, 1, 1);
                this.hoveredNode = null;
            }
            this.hideTooltip();
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
            // Handle click - open node info in new window
            window.open(`/node-info/${node.userData.hash}`, '_blank');
        }
    }
}