export class MatrixInteractions {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.weightPanel = null;
        this.setupWeightPanel();
        this.setupEventListeners();
    }

    setupWeightPanel() {
        this.weightPanel = document.createElement('div');
        this.weightPanel.style.cssText = `
            position: fixed;
            right: 20px;
            top: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
            max-height: 80vh;
            overflow-y: auto;
            font-family: Arial, sans-serif;
            z-index: 1000;
        `;
        document.body.appendChild(this.weightPanel);
    }

    updateWeightPanel(row, col) {
        if (row === -1 || col === -1) {
            this.weightPanel.style.display = 'none';
            return;
        }

        const selectedTag = this.visualizer.tagList[row];
        const weights = [];

        // Get row weights (outgoing relationships)
        this.visualizer.tagList.forEach((toTag, j) => {
            const weight = this.visualizer.tagPairs[selectedTag]?.[toTag] ?? 0;
            if (weight > 0) {
                weights.push({ from: selectedTag, to: toTag, weight });
            }
        });

        // Get column weights (incoming relationships)
        this.visualizer.tagList.forEach((fromTag, i) => {
            if (fromTag === selectedTag) return; // Skip self
            const weight = this.visualizer.tagPairs[fromTag]?.[selectedTag] ?? 0;
            if (weight > 0) {
                weights.push({ from: fromTag, to: selectedTag, weight });
            }
        });

        // Sort by weight descending
        weights.sort((a, b) => b.weight - a.weight);

        // Update panel content
        this.weightPanel.innerHTML = `
            <h3 style="margin-top: 0">Tag Relationships</h3>
            <h4>Selected: ${selectedTag}</h4>
            <div style="margin-bottom: 10px;">
                <strong>Total Relationships:</strong> ${weights.length}
            </div>
            <div style="display: grid; gap: 8px;">
                ${weights.map(({ from, to, weight }) => `
                    <div style="
                        padding: 8px;
                        background: ${from === selectedTag ? '#ffebee' : '#e3f2fd'};
                        border-radius: 4px;
                    ">
                        ${from} → ${to}: ${weight.toFixed(3)}
                    </div>
                `).join('')}
            </div>
        `;
        this.weightPanel.style.display = 'block';
    }

    setupEventListeners() {
        // Mouse events for matrix interaction
        this.visualizer.renderer.domElement.addEventListener('click', (event) => {
            const cell = this.visualizer.getIntersectedCell(event);
            if (cell) {
                const { row, col } = cell;
                if (row === this.visualizer.selectedRow && col === this.visualizer.selectedCol) {
                    // Clicking the same cell again deselects it
                    this.visualizer.clearSelection();
                    this.updateWeightPanel(-1, -1);
                } else {
                    this.visualizer.setSelection(row, col);
                    this.updateWeightPanel(row, col);
                }
            }
        });

        // Click outside to deselect
        document.addEventListener('click', (event) => {
            if (event.target === this.visualizer.renderer.domElement) return;
            if (event.target === this.weightPanel || this.weightPanel.contains(event.target)) return;
            this.visualizer.clearSelection();
            this.updateWeightPanel(-1, -1);
        });

        // Keyboard events
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.visualizer.clearSelection();
                this.updateWeightPanel(-1, -1);
            }
        });

        // Pan controls
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };

        this.visualizer.renderer.domElement.addEventListener('mousedown', (event) => {
            isDragging = true;
            previousMousePosition = {
                x: event.clientX,
                y: event.clientY
            };
        });

        document.addEventListener('mousemove', (event) => {
            if (!isDragging) return;

            const deltaX = event.clientX - previousMousePosition.x;
            const deltaY = event.clientY - previousMousePosition.y;

            const moveSpeed = (this.visualizer.camera.right - this.visualizer.camera.left) 
                            / this.visualizer.renderer.domElement.clientWidth;

            this.visualizer.camera.position.x -= deltaX * moveSpeed;
            this.visualizer.camera.position.y += deltaY * moveSpeed;

            previousMousePosition = {
                x: event.clientX,
                y: event.clientY
            };
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Zoom controls
        this.visualizer.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();
            
            const zoomSpeed = 1.1;
            const zoom = event.deltaY < 0 ? 1 / zoomSpeed : zoomSpeed;
            
            this.visualizer.camera.left *= zoom;
            this.visualizer.camera.right *= zoom;
            this.visualizer.camera.top *= zoom;
            this.visualizer.camera.bottom *= zoom;
            
            this.visualizer.camera.updateProjectionMatrix();
        });
    }
}
