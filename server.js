// server.js
const express = require('express');
const path = require('path');
const HashMapper = require('./hashMapper');
const fs = require('fs').promises;

const app = express();
const port = 3000;

// Define base paths for images
const IMAGE_PATHS = [
    '/mnt/hanoi_data/storage/redpills',
    '/home/offipso/Downloads/redpills'
];

// Initialize HashMapper with the base paths
const hashMapper = new HashMapper({
    basePaths: IMAGE_PATHS,
    cacheFile: path.join(__dirname, 'hash_path_cache.json'),
    updateInterval: 12 * 60 * 60 * 1000 // 12 hours
});

// Initialize the hash mapper when the server starts
hashMapper.initialize().catch(error => {
    console.error('Failed to initialize hash mapper:', error);
});

app.use('/node_modules/three', express.static(path.join(__dirname, 'node_modules/three')));

app.use(express.static(path.join(__dirname, 'public'), {
    setHeaders: (res, path, stat) => {
        if (path.endsWith('.js')) {
            res.set('Content-Type', 'application/javascript; charset=UTF-8');
        }
    }
}));

// Custom middleware to serve images from multiple directories
app.use('/images', (req, res, next) => {
    const imagePath = req.path;

    // Try each base path until we find the image
    const tryNextPath = async (index) => {
        if (index >= IMAGE_PATHS.length) {
            // If we've tried all paths and found nothing, go to next middleware
            return next();
        }

        const fullPath = path.join(IMAGE_PATHS[index], imagePath);

        // Check if file exists in this path
        try {
            await fs.access(fullPath);
            // File exists, serve it
            res.sendFile(fullPath, (err) => {
                if (err) {
                    // If there's an error serving the file, try next path
                    tryNextPath(index + 1);
                }
            });
        } catch (err) {
            // File doesn't exist in this path, try next one
            console.log(`File not found at ${fullPath}`);
            tryNextPath(index + 1);
        }
    };

    // Start trying paths from index 0
    tryNextPath(0);
});

// New endpoint to get path from hash, returning a direct usable URL
app.get('/hash-to-path/:hash', async (req, res) => {
    try {
        await hashMapper.ensureUpdated();
        const filePath = hashMapper.getPath(req.params.hash);

        if (!filePath) {
            res.status(404).json({ error: 'Hash not found' });
            return;
        }

        // Convert filesystem path to web path
        let webPath = filePath;
        for (const basePath of IMAGE_PATHS) {
            if (filePath.startsWith(basePath)) {
                webPath = '/images' + filePath.substring(basePath.length);
                break;
            }
        }

        // Return the full URL including protocol and host
        const baseUrl = `http://localhost:${port}`;
        res.json({ url: `${baseUrl}${webPath}` });
    } catch (error) {
        console.error('Error in hash-to-path endpoint:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Force cache update endpoint (protected by simple API key)
app.post('/update-cache', express.json(), (req, res) => {
    const apiKey = req.headers['x-api-key'];
    if (apiKey !== process.env.HASH_MAPPER_API_KEY) {
        res.status(401).json({ error: 'Unauthorized' });
        return;
    }

    hashMapper.updateCache()
        .then(() => res.json({ message: 'Cache update initiated' }))
        .catch(error => {
            console.error('Error updating cache:', error);
            res.status(500).json({ error: 'Failed to update cache' });
        });
});

// Existing endpoints
app.get('/data', (req, res) => {
    res.sendFile(path.join(__dirname, 'data', 'images_captioned_tagged.json'));
});

app.get('/tag_pairs_with_weights', (req, res) => {
    res.sendFile(path.join(__dirname, 'data', 'tag_pairs_with_weights.json'));
});

app.get('/tags', (req, res) => {
    res.sendFile(path.join(__dirname, 'data', 'tags_with_sizes.json'));
});

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});

app.get('/matrix', (req, res) => {
      res.sendFile(path.join(__dirname, 'public', 'matrix.html'));
});


app.get('/spicy', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'spicy.html'));
});

app.get('/node-info/:hash', (req, res) => {
    const hash = req.params.hash;

    // Read the data file
    fs.readFile(path.join(__dirname, 'data', 'images_captioned_tagged.json'), 'utf8')
        .then(data => {
            const jsonData = JSON.parse(data);
            const nodeData = jsonData[hash];

            if (!nodeData) {
                res.status(404).send('Node not found');
                return;
            }

            // Generate an HTML page with the node information
            const html = `
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Node Information</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }
                        .image {
                            max-width: 100%;
                            height: auto;
                            margin: 20px 0;
                        }
                        .tags {
                            display: flex;
                            flex-wrap: wrap;
                            gap: 8px;
                            margin: 20px 0;
                        }
                        .tag {
                            background: #f0f0f0;
                            padding: 4px 8px;
                            border-radius: 4px;
                        }
                    </style>
                </head>
                <body>
                    <h1>Node Information</h1>
                    <div class="image">
                        <img src="/images/${nodeData.filename}" alt="Node image">
                    </div>
                    <h2>Description</h2>
                    <p>${nodeData.description}</p>
                    <h2>Spiciness Level</h2>
                    <p>${nodeData.spicy.toFixed(2)}</p>
                    <h2>Tags</h2>
                    <div class="tags">
                        ${Object.entries(nodeData.tags)
                .map(([tag, weight]) =>
                    `<div class="tag">${tag} (${weight.toFixed(2)})</div>`)
                .join('')}
                    </div>
                </body>
                </html>
            `;

            res.send(html);
        })
        .catch(err => {
            console.error('Error reading node data:', err);
            res.status(500).send('Internal server error');
        });
});
