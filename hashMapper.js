// hashMapper.js
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class HashMapper {
    constructor(options = {}) {
        this.basePaths = options.basePaths || [];
        this.cacheFile = options.cacheFile || 'hash_path_cache.json';
        this.validExtensions = options.validExtensions || ['.jpg', '.jpeg', '.png', '.gif'];
        this.cache = new Map();
        this.lastUpdate = 0;
        this.updateInterval = options.updateInterval || 24 * 60 * 60 * 1000; // 24 hours
        this.isUpdating = false;
    }

    async initialize() {
        try {
            await this.loadCache();
            if (this.shouldUpdate()) {
                await this.updateCache();
            }
        } catch (error) {
            console.error('Error initializing HashMapper:', error);
            // If cache loading fails, force an update
            await this.updateCache();
        }
    }

    async loadCache() {
        try {
            const data = await fs.readFile(this.cacheFile, 'utf-8');
            const cacheData = JSON.parse(data);
            this.cache = new Map(Object.entries(cacheData.mappings));
            this.lastUpdate = cacheData.lastUpdate;
        } catch (error) {
            if (error.code !== 'ENOENT') {
                console.error('Error loading cache:', error);
            }
            // If file doesn't exist or other error, initialize empty cache
            this.cache = new Map();
            this.lastUpdate = 0;
        }
    }

    async saveCache() {
        const cacheData = {
            lastUpdate: this.lastUpdate,
            mappings: Object.fromEntries(this.cache)
        };
        await fs.writeFile(this.cacheFile, JSON.stringify(cacheData, null, 2));
    }

    shouldUpdate() {
        return Date.now() - this.lastUpdate > this.updateInterval;
    }

    async calculateFileHash(filePath) {
        const fileBuffer = await fs.readFile(filePath);
        return crypto.createHash('sha256').update(fileBuffer).digest('hex');
    }

    async processFile(filePath) {
        try {
            const hash = await this.calculateFileHash(filePath);
            this.cache.set(hash, filePath);
            return { hash, path: filePath };
        } catch (error) {
            console.error(`Error processing file ${filePath}:`, error);
            return null;
        }
    }

    async* walkDirectory(dir) {
        const files = await fs.readdir(dir, { withFileTypes: true });
        for (const file of files) {
            const fullPath = path.join(dir, file.name);
            if (file.isDirectory()) {
                yield* this.walkDirectory(fullPath);
            } else if (this.validExtensions.includes(path.extname(file.name).toLowerCase())) {
                yield fullPath;
            }
        }
    }

    async updateCache() {
        if (this.isUpdating) {
            return;
        }

        this.isUpdating = true;
        const newCache = new Map();
        const startTime = Date.now();

        try {
            for (const basePath of this.basePaths) {
                for await (const filePath of this.walkDirectory(basePath)) {
                    const result = await this.processFile(filePath);
                    if (result) {
                        newCache.set(result.hash, result.path);
                        console.log(`Processed file: ${result.path}`);
                    }
                }
            }

            this.cache = newCache;
            this.lastUpdate = startTime;
            await this.saveCache();
        } catch (error) {
            console.error('Error updating cache:', error);
        } finally {
            this.isUpdating = false;
        }
    }

    getPath(hash) {
        return this.cache.get(hash);
    }

    async ensureUpdated() {
        if (this.shouldUpdate()) {
            await this.updateCache();
        }
    }
}

module.exports = HashMapper;