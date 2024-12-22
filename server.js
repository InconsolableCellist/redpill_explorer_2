const express = require('express');
const path = require('path');
const HashMapper = require('./hashMapper');

const app = express();
const port = 3000;

const hashMapper = new HashMapper({
  basePaths: [
    '/mnt/hanoi_data/storage/redpills',
    '/home/offipso/Downloads/redpills'
  ],
  cacheFile: path.join(__dirname, 'hash_path_cache.json'),
  updateInterval: 12 * 60 * 60 * 1000 // 12 hours
});

// Initialize the hash mapper when the server starts
hashMapper.initialize().catch(error => {
  console.error('Failed to initialize hash mapper:', error);
});

app.use(express.static(path.join(__dirname, 'public')));

app.get('/hash-to-path/:hash', async (req, res) => {
    try {
        await hashMapper.ensureUpdated();
        const filePath = hashMapper.getPath(req.params.hash);

        if (!filePath) {
            res.status(404).json({ error: 'Hash not found' });
            return;
        }

        res.json({ path: filePath });
    } catch (error) {
        console.error('Error in hash-to-path endpoint:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

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
            tryNextPath(index + 1);
        }
    };

    // Start trying paths from index 0
    tryNextPath(0);
});

// New endpoint to get path from hash, now returning web-friendly path
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

        res.json({ path: webPath });
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
