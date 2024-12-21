let width = window.innerWidth;
let height = window.innerHeight;
const unselectedNodeFontSize = 100;
const selectedNodeFontSize = 250;
const connectedNodeFontSize = 200;
const color = d3.scaleOrdinal(d3.schemeCategory10);
const minNodeSize = 50;
const maxNodeSize = 500;
const minLinkWeight = 0.01;
const maxLinkWeight = 5;
const minWeightThreshold = .75;
const mutedColor = '#ddd';  // Light gray

const linkWidthScale = d3.scaleLinear()
    .domain([minWeightThreshold, 1])
    .range([10, 100]);

const linkColorScale = d3.scaleLinear()
  .domain([minWeightThreshold, 1])  // Input: weight range from min threshold to 1
  .range(['red', 'purple']);  // Output: color range from red to purple

let svg;
let tagsWithSizes;
let data;
let simulation;
let tagWeights;
let selectedNodes = [];

window.addEventListener('resize', () => {
  width = window.innerWidth;
  height = window.innerHeight;

  // Update the SVG dimensions
  svg.attr('width', width).attr('height', height);

  if (simulation) {
      simulation.force('center', d3.forceCenter(width / 2, height / 2));
      simulation.alpha(0.3).restart();
  }
});

function sortConnectedNodes(connectedNodes, links, selectedNode) {
  return connectedNodes.map(node => {
    // Find the corresponding link for the weight
    const link = links.find(link =>
      (link.source.id === selectedNode.id && link.target.id === node.id) ||
      (link.target.id === selectedNode.id && link.source.id === node.id)
    );
    return { node, weight: link.weight };
  }).sort((a, b) => b.weight - a.weight);  // Sort by weight descending
}

function setupSVG() {
  const svg = d3.select('#graph')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  // Add a background rectangle to detect deselect clicks
  svg.append('rect')
    .attr('width', width)
    .attr('height', height)
    .attr('fill', 'transparent')
    .attr('class', 'background')
    .on('click', () => {
      // Deselect the current node when background is clicked
      deselectNode();
    });

  return svg;
}

function deselectNode() {
  const link = d3.selectAll('.link');
  const node = d3.selectAll('.node');

  link.style('visibility', 'hidden')
    .attr('opacity', 0);

  node.select('circle').attr('fill', d => color(d.group));

  node.select('text')
    .style('fill', '#000')
    .style('font-size', `${unselectedNodeFontSize}px`)
    .style('font-weight', 'normal');

  const panel = document.getElementById('connectedNodesPanel');
  panel.style.display = 'none';
}

function createForceSimulation(nodes, links) {
  return d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(10000).strength(d => d.weight))
    .force('charge', d3.forceManyBody().strength(-800))  // Repulsive force between nodes
    .force('center', d3.forceCenter(width / 2, height / 2))  // Center the graph
    .force('collision', d3.forceCollide().radius(d => d.size + 10));  // Prevent overlapping nodes
}

function drawLinks(container, links) {
  const filteredLinks = links.filter(link => link.weight >= minWeightThreshold);

  return container.selectAll('.link')
    .data(filteredLinks)
    .enter().append('line')
    .attr('class', 'link')
    .attr('stroke', d => linkColorScale(d.weight))
    .attr('stroke-width', d => linkWidthScale(d.weight))
    .attr('opacity', 0)  // Start with hidden links
    .style('visibility', 'hidden');  // Initially hide links
}

function drawNodes(container, nodes, link, data) {
  const node = container.selectAll('.node')
    .data(nodes)
    .enter().append('g')
    .attr('class', 'node');

  // Append circles for the nodes
  node.append('circle')
    .attr('r', d => d.size)
    .attr('fill', d => color(d.group));

  // Append text labels for the nodes
  node.append('text')
    .text(d => `${d.id} (${d.itemIds.length})`)
    .attr('x', 0)
    .attr('y', d => -d.size - 5)
    .attr('text-anchor', 'middle')
    .style('font-size', '${unselectedNodeFontSize}px')
    .style('font-weight', 'normal');

  return node;
}

// Function to prepare the graph by filtering nodes and links
function prepareGraph(tagsWithSizes, tagWeights) {
  const nodes = Object.keys(tagsWithSizes).map((tag, index) => {
    const tagData = tagsWithSizes[tag];
    const size = minNodeSize + tagData.normalizedSize * (maxNodeSize - minNodeSize);
    return {
      id: tag,
      type: 'tag',
      group: index,
      itemIds: tagData.itemIds,
      size: size,
      normalizedSize: tagData.normalizedSize
    };
  });

  const nodeMap = {};
  nodes.forEach(node => {
    nodeMap[node.id] = node;
  });

  const links = [];
  for (const tag1 in tagWeights) {
    for (const tag2 in tagWeights[tag1]) {
      if (nodeMap[tag1] && nodeMap[tag2] && tagWeights[tag1][tag2] >= minWeightThreshold) {
        links.push({
          source: nodeMap[tag1],
          target: nodeMap[tag2],
          weight: tagWeights[tag1][tag2]
        });
      }
    }
  }

  return { filteredNodes: nodes, filteredLinks: links };
}

function updateConnectedNodesPanel(node, links, allNodes) {
  const connectedNodes = getConnectedNodes(node, links);
  const panel = document.getElementById('connectedNodesPanel');

  if (connectedNodes.length > 0) {
    // Sort connected nodes by link weight
    const sortedConnectedNodes = sortConnectedNodes(connectedNodes, links, node);

    panel.innerHTML = `<strong>Connected to ${node.id}:</strong><ul>` +
      sortedConnectedNodes.map(({ node, weight }) =>
        `<li><a href="#" class="connected-node" data-node-id="${node.id}">${node.id} (${weight.toFixed(2)})</a></li>`
      ).join('') +
      `</ul>`;
    panel.style.display = 'block';

    // Add click event listeners to each node name in the panel
    document.querySelectorAll('.connected-node').forEach(el => {
      el.addEventListener('click', (event) => {
        event.preventDefault();  // Prevent default link behavior
        const nodeId = event.target.getAttribute('data-node-id');
        const targetNode = allNodes.find(n => n.id === nodeId);  // Find the node by its ID
        if (targetNode) {
          highlightNode(targetNode, links, allNodes);  // Highlight the clicked node
          updateConnectedNodesPanel(targetNode, links, allNodes);  // Update the panel for the new node
        }
      });
    });
  } else {
    panel.style.display = 'none';  // Hide the panel if no connections
  }
}

// Function to get the connected nodes for a selected node
function getConnectedNodes(node, links) {
  return links
    .filter(link => link.source.id === node.id || link.target.id === node.id)
    .map(link => (link.source.id === node.id ? link.target : link.source));
}
function isConnected(node, clickedNode, links) {
  return links.some(d => (d.source.id === clickedNode.id && d.target.id === node.id) ||
                         (d.target.id === clickedNode.id && d.source.id === node.id));
}

function highlightNode(clickedNode, links, allNodes) {
  const link = d3.selectAll('.link');
  const node = d3.selectAll('.node');

  // Show related links and nodes
  link.style('visibility', d => (d.source.id === clickedNode.id || d.target.id === clickedNode.id) ? 'visible' : 'hidden')
    .attr('stroke', d => (d.source.id === clickedNode.id || d.target.id === clickedNode.id)
      ? linkColorScale(d.weight)
      : mutedColor)
    .attr('stroke-width', d => (d.source.id === clickedNode.id || d.target.id === clickedNode.id)
      ? linkWidthScale(d.weight)
      : 0)
    .attr('stroke-opacity', d => (d.source.id === clickedNode.id || d.target.id === clickedNode.id) ? 1 : 0.5)
    .attr('opacity', d => (d.source.id === clickedNode.id || d.target.id === clickedNode.id) ? 1 : 0);

  // Highlight the selected node and adjust font size and boldness
  node.select('circle')
    .attr('fill', d => (d.id === clickedNode.id || isConnected(d, clickedNode, links))
      ? color(d.group)
      : mutedColor);

  node.select('text')
    .style('fill', d => (d.id === clickedNode.id || isConnected(d, clickedNode, links))
      ? '#000'
      : mutedColor)
    .style('font-size', d => d.id === clickedNode.id
      ? `${selectedNodeFontSize}px`
      : isConnected(d, clickedNode, links)
        ? `${connectedNodeFontSize}px`
        : `${unselectedNodeFontSize}px`)
    .style('font-weight', d => d.id === clickedNode.id ? 'bold' : 'normal');
}


function openItemPanel(node) {
    const panel = document.getElementById('itemPanel');
    const closeBtn = panel.querySelector('.close-btn');
    const itemGrid = panel.querySelector('.item-grid');
    const tagArea = panel.querySelector('.tag-area');

    // Replace the tag input and tag list with a container for better layout control
    let filterContainer = panel.querySelector('.filter-container');
    if (!filterContainer) {
        filterContainer = document.createElement('div');
        filterContainer.className = 'filter-container';
        panel.insertBefore(filterContainer, itemGrid);
    } else {
        filterContainer.innerHTML = '';
    }

    // Create the tag input and tag list
    let tagInput = document.createElement('input');
    tagInput.type = 'text';
    tagInput.className = 'tag-input';
    tagInput.placeholder = 'Type to filter tags...';

    let tagList = document.createElement('div');
    tagList.className = 'tag-list';

    // Create the caption filter input
    let captionInput = document.createElement('input');
    captionInput.type = 'text';
    captionInput.className = 'caption-input';
    captionInput.placeholder = 'Type to filter captions...';

    // Append the inputs and tag list to the filter container
    const tagFilterWrapper = document.createElement('div');
    tagFilterWrapper.className = 'filter-wrapper';
    tagFilterWrapper.appendChild(tagInput);
    tagFilterWrapper.appendChild(tagList);

    const captionFilterWrapper = document.createElement('div');
    captionFilterWrapper.className = 'filter-wrapper';
    captionFilterWrapper.appendChild(captionInput);

    // Create the recommended tags container
    let recommendedTagsContainer = document.createElement('div');
    recommendedTagsContainer.className = 'recommended-tags-container';

    // Append the recommended tags container under captionInput
    captionFilterWrapper.appendChild(recommendedTagsContainer);

    filterContainer.appendChild(tagFilterWrapper);
    filterContainer.appendChild(captionFilterWrapper);

    let selectedTags = [node.id];
    let tagsInItems = []; // To store tags with counts

    // Function to get combined recommended tags for all selected tags
    function getCombinedRecommendedTags(selectedTags) {
        const recommendationsMap = {};

        selectedTags.forEach(tag => {
            const relatedTags = tagWeights[tag] || {};
            for (const [relatedTag, weight] of Object.entries(relatedTags)) {
                if (weight >= minWeightThreshold && !selectedTags.includes(relatedTag)) {
                    if (recommendationsMap[relatedTag]) {
                        // Combine weights by taking the maximum
                        recommendationsMap[relatedTag] = Math.max(recommendationsMap[relatedTag], weight);
                    } else {
                        recommendationsMap[relatedTag] = weight;
                    }
                }
            }
        });

        // Convert to array and sort by weight descending
        const recommendations = Object.entries(recommendationsMap)
            .map(([tag, weight]) => ({ tag, weight }))
            .sort((a, b) => b.weight - a.weight);

        return recommendations;
    }

    function updateItemPanel() {
        itemGrid.innerHTML = '';
        tagArea.innerHTML = '';
        tagList.innerHTML = ''; // Clear the tag list

        // Display selected tags at the top
        selectedTags.forEach(tag => {
            const tagElem = document.createElement('span');
            tagElem.className = 'selected-tag';
            tagElem.textContent = tag + ' ×';
            tagElem.addEventListener('click', () => {
                selectedTags = selectedTags.filter(t => t !== tag);
                updateItemPanel();
            });
            tagArea.appendChild(tagElem);
        });

        // Get items matching all selected tags
        const items = Object.keys(data).map(id => {
            const item = data[id];
            item.tags = Object.keys(item.tags);
            return { id, ...item };
        }).filter(item => {
            return selectedTags.every(tag => item.tags && item.tags.includes(tag));
        }).filter(item => {
            // Filter by caption text if provided
            const captionFilterText = captionInput.value.trim().toLowerCase();
            if (captionFilterText) {
                const caption = item.description && item.description['<MORE_DETAILED_CAPTION>'] || '';
                return caption.toLowerCase().includes(captionFilterText);
            }
            return true;
        });

        // Collect tags from the currently displayed items with counts
        const tagCounts = {};
        items.forEach(item => {
            if (item.tags) {
                item.tags.forEach(tag => {
                    if (!selectedTags.includes(tag)) {
                        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
                    }
                });
            }
        });

        // Convert tagCounts to an array and sort it by counts descending
        tagsInItems = Object.entries(tagCounts).sort((a, b) => b[1] - a[1]);

        // Update the tag list
        updateTagList();

        // Update the recommended tags list
        updateRecommendedTags();

        // Sort items by spicy level
        items.sort((a, b) => parseInt(b.spicy) - parseInt(a.spicy));

        // Limit the number of items displayed to prevent browser crash
        const MAX_ITEMS = 100;
        let displayedItems = items;
        // TODO: the styling is kind of shit, it appears inline with the thumbnails
        /*
        if (items.length > MAX_ITEMS) {
            displayedItems = items.slice(0, MAX_ITEMS);
            // Display a message indicating truncation
            const message = document.createElement('p');
            message.textContent = `Showing first ${MAX_ITEMS} items out of ${items.length}.`;
            itemGrid.appendChild(message);
        }*/

        // Check if there are items to display
        if (displayedItems.length > 0) {
            // For each item, create a card
            displayedItems.forEach(item => {
                const card = document.createElement('div');
                card.className = 'item-card';

                // Get an excerpt from <MORE_DETAILED_CAPTION>
                const caption = item.description && item.description['<MORE_DETAILED_CAPTION>'] || '';
                const excerpt = caption.length > 750 ? caption.substring(0, 750) + '...' : caption;

                // Spicy level
                const spicyLevel = item.spicy || '0';

                // Get the tags and truncate if necessary
                const tags = item.tags || [];
                const tagsText = tags.join(', ');
                const maxTagsLength = 100; // Maximum characters to display
                const displayedTags = tagsText.length > maxTagsLength ? tagsText.substring(0, maxTagsLength) + '...' : tagsText;

                // Set background image to thumbnail
                const hash = item.id;
                const thumbnailUrl = `/thumbnails/${hash[0]}/${hash[1]}/${hash[2]}/${hash.substring(3)}.jpg`;
                card.style.backgroundImage = `url('${thumbnailUrl}')`;

                // Create card content
                card.innerHTML = `
                    <div class="item-content" title="Tags: ${tagsText}">
                        <p class="spicy-level"><strong>Spicy Level:</strong> ${spicyLevel}</p>
                        <p>${excerpt}</p>
                        <p class="tags">${displayedTags}</p>
                    </div>
                `;

                // Add click event to open image
                card.addEventListener('click', () => {
                    const imagePath = item.path || '';
                    let imageUrl = imagePath;
                    imageUrl = imageUrl.replace('/mnt/hanoi_data/storage/redpills', '/images/');
                    imageUrl = imageUrl.replace('/home/offipso/Downloads/redpills', '/images/');
                    imageUrl = imageUrl.replace(/\/+/g, '/'); // Replace multiple slashes with a single slash
                    const encodedUrl = encodeURI(imageUrl);
                    window.open(encodedUrl, '_blank');
                });

                itemGrid.appendChild(card);
            });
        } else {
            // Display a message when no items match
            const message = document.createElement('p');
            message.textContent = 'No items match the selected filters.';
            itemGrid.appendChild(message);
        }
    }

    // Function to update the tag list based on input
    function updateTagList() {
        tagList.innerHTML = '';
        const filterText = tagInput.value.trim().toLowerCase();
        let filteredTags = tagsInItems;

        if (filterText) {
            filteredTags = filteredTags.filter(([tag, count]) => tag.toLowerCase().includes(filterText));
        }

        if (filteredTags.length > 0) {
            filteredTags.forEach(([tag, count]) => {
                const tagItem = document.createElement('div');
                tagItem.className = 'tag-item';
                tagItem.textContent = `${tag} (${count})`;
                tagItem.addEventListener('click', () => {
                    if (!selectedTags.includes(tag)) {
                        selectedTags.push(tag);
                        updateItemPanel();
                    }
                });
                tagList.appendChild(tagItem);
            });
        } else {
            const noTag = document.createElement('div');
            noTag.className = 'no-tag';
            noTag.textContent = 'No tags found';
            tagList.appendChild(noTag);
        }
    }

    // Function to update the recommended tags list
    function updateRecommendedTags() {
        recommendedTagsContainer.innerHTML = ''; // Clear previous content

        const recommendedTags = getCombinedRecommendedTags(selectedTags);

        if (recommendedTags.length > 0) {
            // Create a title for the recommended tags section
            const title = document.createElement('h3');
            title.textContent = 'Recommended Tags';
            recommendedTagsContainer.appendChild(title);

            // Limit to 5 items visible at a time with scrolling
            const recommendedTagsList = document.createElement('div');
            recommendedTagsList.className = 'recommended-tags-list';

            recommendedTags.forEach(({ tag, weight }) => {
                const tagItem = document.createElement('div');
                tagItem.className = 'recommended-tag-item';
                tagItem.textContent = `${tag} (${weight.toFixed(2)})`;
                tagItem.addEventListener('click', () => {
                    if (!selectedTags.includes(tag)) {
                        selectedTags.push(tag);
                        updateItemPanel();
                    }
                });
                recommendedTagsList.appendChild(tagItem);
            });

            recommendedTagsContainer.appendChild(recommendedTagsList);
        } else {
            const noRecommendations = document.createElement('div');
            noRecommendations.className = 'no-recommendations';
            noRecommendations.textContent = 'No recommended tags.';
            recommendedTagsContainer.appendChild(noRecommendations);
        }
    }

    // Event listener for tag input field
    tagInput.addEventListener('input', updateTagList);

    // Event listener for caption input field
    captionInput.addEventListener('input', () => {
        updateItemPanel();
    });

    // Build initial panel
    updateItemPanel();

    // Show the panel
    panel.style.display = 'block';

    // Close button event
    closeBtn.onclick = () => {
        closeItemPanel();
    };
}

function closeItemPanel() {
    const panel = document.getElementById('itemPanel');
    panel.style.display = 'none';
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        const panel = document.getElementById('itemPanel');
        if (panel.style.display === 'block') {
            closeItemPanel();
        }
    }
});

function toggleNodeSelection(node) {
  const index = selectedNodes.findIndex(n => n.id === node.id);
  if (index >= 0) {
    selectedNodes.splice(index, 1); // Remove node from selection
  } else {
    selectedNodes.push(node); // Add node to selection
  }
}

function highlightSelectedNodes(links, allNodes) {
  const link = d3.selectAll('.link');
  const node = d3.selectAll('.node');

  // Reset styles for all nodes and links
  link.style('visibility', 'hidden').attr('opacity', 0);

  node.select('circle').attr('fill', mutedColor);

  node.select('text')
    .style('fill', mutedColor)
    .style('font-size', `${unselectedNodeFontSize}px`)
    .style('font-weight', 'normal');

  // Highlight selected nodes and their connections
  selectedNodes.forEach(selectedNode => {
    // Highlight the selected node
    node.filter(d => d.id === selectedNode.id).select('circle')
      .attr('fill', d => color(d.group));

    node.filter(d => d.id === selectedNode.id).select('text')
      .style('fill', '#000')
      .style('font-size', `${selectedNodeFontSize}px`)
      .style('font-weight', 'bold');

    // Show and style links connected to the selected node
    link.filter(d => d.source.id === selectedNode.id || d.target.id === selectedNode.id)
      .style('visibility', 'visible')
      .attr('stroke', d => linkColorScale(d.weight))
      .attr('stroke-width', d => linkWidthScale(d.weight))
      .attr('stroke-opacity', 1)
      .attr('opacity', 1);

    // Highlight connected nodes
    node.filter(d => isConnected(d, selectedNode, links))
      .select('circle')
      .attr('fill', d => color(d.group));

    node.filter(d => isConnected(d, selectedNode, links))
      .select('text')
      .style('fill', '#000')
      .style('font-size', `${connectedNodeFontSize}px`)
      .style('font-weight', 'normal');
  });
}

function fetchAndDisplayItems(selectedNodes) {
  const selectedTags = selectedNodes.map(node => node.id);
  if (selectedTags.length > 0) {
    const tagsText = selectedTags.join(',');

    // Make HTTP POST request to server
    const url = 'http://bestiary:8000/search?k=5&store_in_db=false';
    const formData = new FormData();
    formData.append('file', '');
    formData.append('image_path', 'string');
    formData.append('image_hash', 'string');
    formData.append('text', tagsText);

    fetch(url, {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // 'data' contains the response from the server
      // Display items in the new view
      displayItems(data.results);
    })
    .catch(error => {
      console.error('Error fetching items:', error);
    });
  } else {
    // No tags selected, clear the item view
    clearItemView();
  }
}

function displayItems(items) {
  const itemView = document.getElementById('itemView');
  itemView.innerHTML = ''; // Clear previous items

  items.forEach(item => {
    const itemNode = document.createElement('div');
    itemNode.className = 'item-node';

    // Assuming item has 'id' and 'description' properties
    const hash = item.hash;
    const thumbnailUrl = `/thumbnails/${hash[0]}/${hash[1]}/${hash[2]}/${hash.substring(3)}.jpg`;

    itemNode.style.backgroundImage = `url('${thumbnailUrl}')`;

    const captionElem = document.createElement('div');
    captionElem.className = 'item-caption';
    captionElem.textContent = item.description || '';

    itemNode.appendChild(captionElem);

    // Add click handler to open item panel
    itemNode.addEventListener('click', () => {
      openItemPanel(item);
    });

    itemView.appendChild(itemNode);
  });
}

function clearItemView() {
  const itemView = document.getElementById('itemView');
  itemView.innerHTML = '';
}

// Load tags, data, and weights
Promise.all([d3.json('/tags'), d3.json('/data'), d3.json('/tag_pairs_with_weights')])
  .then(([tagsWithSizesLoaded, dataLoaded, tagWeightsLoaded]) => {
    // Create force-directed graph layout
    tagsWithSizes = tagsWithSizesLoaded;
    tagWeights = tagWeightsLoaded
    data = dataLoaded;
    const { filteredNodes, filteredLinks } = prepareGraph(tagsWithSizes, tagWeights);
    svg = setupSVG();
    const container = svg.append('g');  // Group for nodes and links

    // Add zoom behavior
    svg.call(d3.zoom()
      .scaleExtent([0.01, 10])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      }));

    svg.select('.background').on('click', () => {
      selectedNodes = [];
      highlightSelectedNodes(filteredLinks, filteredNodes);
      fetchAndDisplayItems(selectedNodes);
    });

    const simulation = createForceSimulation(filteredNodes, filteredLinks);

    const link = drawLinks(container, filteredLinks);

    const node = drawNodes(container, filteredNodes, link, data);

    node.on('click', (event, d) => {
        toggleNodeSelection(d);
        highlightSelectedNodes(filteredLinks, filteredNodes);
        fetchAndDisplayItems(selectedNodes);
        event.stopPropagation();
    });

    node.on('dblclick', (event, d) => {
        openItemPanel(d);
        event.stopPropagation();
    });

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    simulation.on('end', () => {
      simulation.stop();
    });

  }).catch(error => {
    console.error('Error loading data:', error);
    alert('Failed to load data. Please try again later.');
  });
