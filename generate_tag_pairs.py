#!/usr/bin/env python3
import json
import itertools

# Configurable minimum number of entries for tags to be included
minEntries = 10

# Load tags_with_sizes.json
with open('data/tags_with_sizes.json', 'r') as f:
    tags_with_sizes = json.load(f)

# Filter tags with more than minEntries items
filtered_tags = [tag for tag, data in tags_with_sizes.items() if len(data['itemIds']) >= minEntries]

# Generate all possible combinations of tag pairs
# We'll use a dictionary to store the pairs in the format { 'tag1': { 'tag2': weight } }
tag_pairs = {}

for tag1, tag2 in itertools.combinations(filtered_tags, 2):
    # Initialize the nested dictionary if not already present
    if tag1 not in tag_pairs:
        tag_pairs[tag1] = {}
    # Do not add the reverse (tag2 -> tag1) as it's already covered
    tag_pairs[tag1][tag2] = None  # No weight assigned yet

# Save the tag pairs to a JSON file
with open('data/tag_pairs.json', 'w') as f:
    json.dump(tag_pairs, f, indent=2)

print(f"Total unique tag pairs generated: {len(tag_pairs)}")
