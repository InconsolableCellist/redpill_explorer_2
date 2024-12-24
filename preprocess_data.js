// preprocess_data.js

const fs = require('fs');
const path = require('path');

// Configurable minimum number of entries for tags to be included
const minEntries = 5;

// Load the dataset
const dataPath = path.join(__dirname, 'data', 'images_captioned_tagged.json');
const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

// Create a mapping from tags to item IDs
const tagsToItems = {};

for (const itemId in data) {
  const item = data[itemId];
  const tags = item.tags ? Object.keys(item.tags) : [];

  tags.forEach(tag => {
    if (!tagsToItems[tag]) {
      tagsToItems[tag] = [];
    }
    tagsToItems[tag].push(itemId);
  });
}

// Remove tags that have fewer entries than minEntries
for (const tag in tagsToItems) {
  if (tagsToItems[tag].length < minEntries) {
    delete tagsToItems[tag];
  }
}

// Find the maximum and minimum number of items associated with any tag
let maxItemCount = 0;
let minItemCount = Infinity;

for (const tag in tagsToItems) {
  const itemCount = tagsToItems[tag].length;
  if (itemCount > maxItemCount) {
    maxItemCount = itemCount;
  }
  if (itemCount < minItemCount) {
    minItemCount = itemCount;
  }
}

// Handle the case where maxItemCount equals minItemCount
if (maxItemCount === minItemCount) {
  console.warn('All tags have the same number of items. Assigning normalizedSize of 1 to all tags.');
}

// Calculate normalized size for each tag using linear scaling
const tagsWithSizes = {};
for (const tag in tagsToItems) {
  const itemCount = tagsToItems[tag].length;
  let normalizedSize = 1; // Default value

  if (maxItemCount !== minItemCount) {
    normalizedSize = (itemCount - minItemCount) / (maxItemCount - minItemCount);
  }

  tagsWithSizes[tag] = {
    itemIds: tagsToItems[tag],
    normalizedSize: normalizedSize
  };
}

// Save the mapping with sizes to a new JSON file
const tagsWithSizesPath = path.join(__dirname, 'data', 'tags_with_sizes.json');
fs.writeFileSync(tagsWithSizesPath, JSON.stringify(tagsWithSizes, null, 2));

// Adjust image paths and save the modified data
const adjustedData = {};

for (const itemId in data) {
  const item = data[itemId];

  // Update the path to be relative to the public/images directory
  item.relativePath = path.join('images', item.filename);

  adjustedData[itemId] = item;
}

// Save the adjusted data
const adjustedDataPath = path.join(__dirname, 'data', 'adjusted_data.json');
fs.writeFileSync(adjustedDataPath, JSON.stringify(adjustedData, null, 2));

console.log('Data preprocessing completed successfully.');
