const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname, 'public')));

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
