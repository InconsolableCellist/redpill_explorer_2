// spicyInteractions.js
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SpicySliceView } from './spicySliceView.js';

export class SpicyInteractions {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.controls = null;
        this.tooltip = document.getElementById('tooltip');
        this.isTooltipVisible = false;
        this.setupOrbitControls();
        this.setupEventListeners();
        this.setupTagPanel();

        // Initialize the slice view handler
        this.sliceView = new SpicySliceView(visualizer);

        // Camera parameters
        this.cameraRadius = 600;
        this.cameraAngle = Math.PI / 4;
        this.cameraHeight = 500;
        this.rotationAngle = 0;
    }

    setupOrbitControls() {
        this.controls = new OrbitControls(
            this.visualizer.camera,
            this.visualizer.renderer.domElement
        );

        this.controls.enabled = false;
        this.updateCameraPosition();

        let isMouseDown = false;
        let previousX = 0;
        let previousY = 0;

        this.visualizer.renderer.domElement.addEventListener('mousedown', (event) => {
            if (event.button === 0) {
                isMouseDown = true;
                previousX = event.clientX;
                previousY = event.clientY;
            }
        });

        this.visualizer.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;

            const deltaX = event.clientX - previousX;
            const deltaY = event.clientY - previousY;

            this.rotationAngle -= deltaX * 0.01;
            this.cameraHeight -= deltaY * 2;
            this.updateCameraPosition();

            previousX = event.clientX;
            previousY = event.clientY;
        });

        document.addEventListener('mouseup', (event) => {
            if (event.button === 0) {
                isMouseDown = false;
            }
        });

        this.visualizer.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();
            const zoomSpeed = 0.1;
            const delta = event.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;
            this.cameraRadius = Math.max(200, Math.min(1000, this.cameraRadius * delta));
            this.updateCameraPosition();
        });

        this.visualizer.renderer.domElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });
    }

    updateCameraPosition() {
        const horizontalRadius = this.cameraRadius * Math.sin(this.cameraAngle);
        const x = horizontalRadius * Math.cos(this.rotationAngle);
        const z = horizontalRadius * Math.sin(this.rotationAngle);
        const y = this.cameraHeight + this.cameraRadius * Math.cos(this.cameraAngle);

        this.visualizer.camera.position.set(x, y, z);
        this.visualizer.camera.lookAt(0, this.cameraHeight, 0);
        this.controls.target.set(0, this.cameraHeight, 0);
    }

    setupEventListeners() {
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

        this.visualizer.renderer.domElement.addEventListener('click', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (!intersectedObject) return;

            if (intersectedObject.userData.type === 'slice') {
                this.sliceView.showSliceView(intersectedObject);
            } else if (intersectedObject.userData.type === 'node') {
                this.handleNodeClick(intersectedObject);
            }
        });

        this.visualizer.renderer.domElement.addEventListener('dblclick', (event) => {
            const intersectedObject = this.visualizer.getIntersectedNode(event);
            if (intersectedObject && intersectedObject.userData.type === 'node') {
                this.handleNodeDoubleClick(intersectedObject);
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.sliceView.closeSliceView();
                this.visualizer.selectedTags.clear();
                this.updateSelectedTagsPanel();
                this.visualizer.updateSelection();
            }
        });
    }

    updateTooltip(event, node) {
        if (node && node.userData.hash) {
            const itemData = this.visualizer.nodeData.get(node.userData.hash);
            if (itemData) {
                const description = itemData.description.length > 400
                    ? itemData.description.substring(0, 400) + '...'
                    : itemData.description;

                this.tooltip.innerHTML = `
                    <div><strong>Spicy Level:</strong> ${itemData.spicy.toFixed(2)}</div>
                    <div><strong>Description:</strong> ${description}</div>
                    <div><strong>Tags:</strong> ${Object.keys(itemData.tags).join(', ')}</div>
                `;

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
        const response = await fetch('/tags');
        const tagsData = await response.json();

        const tagList = document.getElementById('tagList');
        const tagSearch = document.getElementById('tagSearch');
        const selectedTagsPanel = document.getElementById('selectedTags');

        const sortedTags = Object.entries(tagsData)
            .sort(([, a], [, b]) => b.itemIds.length - a.itemIds.length)
            .map(([tag]) => tag);

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

                    if (this.sliceView.selectedSlice) {
                        this.sliceView.showSliceView(this.sliceView.selectedSlice);
                    }
                }
            });

            tagList.appendChild(tagElement);
        });

        tagSearch.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            Array.from(tagList.children).forEach(tagElement => {
                const tagText = tagElement.textContent.toLowerCase();
                tagElement.style.display = tagText.includes(searchTerm) ? 'block' : 'none';
            });
        });

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

                const tagListItem = Array.from(document.getElementsByClassName('tag-item'))
                    .find(el => el.textContent.includes(tag));
                if (tagListItem) {
                    tagListItem.classList.remove('selected');
                }

                if (this.sliceView.selectedSlice) {
                    this.sliceView.showSliceView(this.sliceView.selectedSlice);
                }
            });

            panel.appendChild(tagElement);
        });
    }

    handleNodeClick(node) {
        window.open(`/node-info/${node.userData.hash}`, '_blank');
    }

    handleNodeDoubleClick(node) {
        window.open(`/node-info/${node.userData.hash}`, '_blank');
    }
}