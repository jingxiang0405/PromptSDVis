function initPromptTree(dataset) {
    const width = 980, height = 400;
    const nodeSpacing = 150; // 节点之间的间距
    const startX = 80;       // 起始X坐标
    const centerY = height / 2; // Y坐标保持居中

    const svg = d3.select("#prompt-tree").append("svg")
        .attr("width", width)
        .attr("height", height);

    // 计算节点的左至右布局位置
    dataset.nodes.forEach((node, i) => {
        node.x = startX + i * nodeSpacing; // 计算每个节点的x坐标
        node.y = centerY; // y坐标固定在中心
    });

    // 绘制连线
    const link = svg.selectAll(".link")
        .data(dataset.links)
        .enter().append("line")
        .attr("class", "link")
        .attr("x1", d => dataset.nodes[d.source].x)
        .attr("y1", d => dataset.nodes[d.source].y)
        .attr("x2", d => dataset.nodes[d.target].x)
        .attr("y2", d => dataset.nodes[d.target].y)
        .style("stroke", "#ccc")
        .style("stroke-width", 2);

    // 绘制文本背景
    const textBackground = svg.selectAll(".text-bg")
        .data(dataset.nodes)
        .enter()
        .append("rect")
        .attr("class", "text-bg");

    // 添加节点标签
    const text = svg.selectAll(".text")
        .data(dataset.nodes)
        .enter().append("text")
        .attr("class", "text")
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .text(d => d.name)
        .each(function (d) {
            const bbox = this.getBBox();
            d.bbox = bbox;
        });

    // 更新文本背景位置和大小
    textBackground
        .attr("x", d => d.bbox.x - 5)
        .attr("y", d => d.bbox.y - 2)
        .attr("width", d => d.bbox.width + 10)
        .attr("height", d => d.bbox.height + 4)
        .attr("fill", "#f0f0f0");

    function updateGraph() {

        const nodes = svg.selectAll(".node")
            .data(dataset.nodes, d => d.id);
        // 处理退出的旧节点
        nodes.exit().remove();

        const enterNodes = nodes.enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x}, ${d.y})`);

        enterNodes.append("text")
            .attr("class", "text")
            .text(d => d.name)
            .each(function (d) {
                const bbox = this.getBBox();
                d.bbox = bbox;
            });

        // 为新节点添加文本背景
        enterNodes.insert("rect", "text")
            .attr("class", "text-bg")
            .attr("x", d => d.bbox.x - 5)
            .attr("y", d => d.bbox.y - 2)
            .attr("width", d => d.bbox.width + 10)
            .attr("height", d => d.bbox.height + 4)
            .attr("fill", "#f0f0f0");
        const links = svg.selectAll(".link")
            .data(dataset.links, d => `${d.source}-${d.target}`);
        console.log('update link')
        console.log(dataset.links)
        links.enter().append("line")
            .attr("class", "link")
            .attr("x1", d => {
                const sourceNode = dataset.nodes.find(node => node.id === d.source);
                return sourceNode.x;
            })
            .attr("y1", d => {
                const sourceNode = dataset.nodes.find(node => node.id === d.source);
                return sourceNode.y;
            })
            .attr("x2", d => {
                const targetNode = dataset.nodes.find(node => node.id === d.target);
                return targetNode.x;
            })
            .attr("y2", d => {
                const targetNode = dataset.nodes.find(node => node.id === d.target);
                return targetNode.y;
            })
            .style("stroke", "#ccc")
            .style("stroke-width", 2);
        // 处理退出的旧链接
        links.exit().remove();

    }
    function removeNewNodesAndLinks(d) {
        //console.log('removeNewNodesAndLinks:' + d.id)

        // 找出所有标记为 'isNew' 且与特定节点 d 相关的节点的 ID
        let newNodeIds = dataset.nodes.filter(node => node.isNew && (node.relatedId === d.id)).map(node => node.id);
        // console.log('newNodeIds:' + newNodeIds)
        // 过滤掉这些节点
        dataset.nodes = dataset.nodes.filter(node => !newNodeIds.includes(node.id));
        // console.log(dataset.links);
        // console.log("Removed Before.");
        // 同时过滤掉与这些节点相关的链接
        dataset.links = dataset.links.filter(link => !newNodeIds.includes(link.source) && !newNodeIds.includes(link.target));
        // console.log("Removed After.");
        // console.log(dataset.links);
        updateGraph()
        // console.log("Removed new nodes and links.");

        // console.log(dataset.nodes);
    }
    // 假设节点的 ID 是从 0 开始，递增
    let lastId = dataset.nodes.reduce((max, node) => node.id > max ? node.id : max, -1);

    // 全局 ID 计数器初始化为最大 ID + 1
    let nodeIdCounter = lastId + 1;

    function addNewNodeBySimilarity(nodes, d) {

        const currentNodeIndex = nodes.findIndex(node => node.id === d.id);
        const currentNode = nodes[currentNodeIndex]; // 當前節點
        const isEnd = currentNode.isEnd;
        const nextNodeIndex = currentNodeIndex + 1; // 假設有順序的下一個節點
        const nextNode = nodes[nextNodeIndex]; // 獲取下一個節點

        const addNodes = (similarityData, yOffset) => {
            similarityData.forEach((data, index) => {
                let newX, newY;
                if (!isEnd) {
                    newX = (d.x + nextNode.x) / 2;
                    newY = d.y + yOffset * (index + 1);
                } else {
                    newX = d.x + nodeSpacing / 2;
                    newY = d.y + yOffset * (index + 1);
                }
                console.log('d.id:' + d.id)
                const newNode = {
                    id: nodeIdCounter++,
                    name: data.name + ':' + ((d.id + 1) * 10 + dataset.nodes.length) + ':' + d.id,
                    x: newX,
                    y: newY,
                    hasSimilarity: false,
                    isNew: true,
                    relatedId: d.id  // 设置关联ID为当前节点的ID
                };
                dataset.nodes.push(newNode);
                dataset.links.push({ source: d.id, target: newNode.id });
                if (!isEnd) {
                    console.log('end')
                    dataset.links.push({ source: newNode.id, target: nextNode.id });
                }
            });
        };

        let diffusiondbSim = currentNode.diffusiondb.map(d => ({ similarity: d.similarity, name: d.name }));
        //let laion4bSim = currentNode.laion4b.map(d => ({ similarity: d.similarity, name: d.name }));

        diffusiondbSim.sort((a, b) => b.similarity - a.similarity);
        //laion4bSim.sort((a, b) => b.similarity - a.similarity);

        addNodes(diffusiondbSim, -50); // 向上添加
        //addNodes(laion4bSim, 50); // 向下添加

        updateGraph();

    }


    // 添加加号圆形和交叉线
    dataset.nodes.filter(d => d.hasSimilarity).forEach((d, i) => {
        const plusGroup = svg.append("g")
            .attr("class", "plus-group")
            .classed("active", d.hasSimilarity)
            .attr("transform", `translate(${d.bbox.x + d.bbox.width + 15}, ${d.y})`)
            .on("click", function () {
                const group = d3.select(this);
                const isActive = group.classed("active");
                console.log(isActive)
                if (isActive) {
                    group.select(".plus-line.vertical").style("opacity", 0);
                    addNewNodeBySimilarity(dataset.nodes, d);
                    console.log('addNewNodeBySimilarity dataset.nodes.filter')
                    console.log(dataset.nodes)
                } else {
                    group.select(".plus-line.vertical").style("opacity", 1);
                    removeNewNodesAndLinks(d);
                    console.log('removeNewNodesAndLinks dataset.nodes.filter')
                    console.log(dataset.nodes)
                }
                // 只更新当前点击的组的状态
                group.classed("active", !isActive);
            });



        plusGroup.append("circle")
            .attr("class", "plus-circle")
            .attr("r", 8);

        plusGroup.append("line")
            .attr("class", "plus-line horizontal")
            .attr("x1", -5)
            .attr("y1", 0)
            .attr("x2", 5)
            .attr("y2", 0);

        // 添加垂直線，初始時為可見（opacity為1）
        plusGroup.append("line")
            .attr("class", "plus-line vertical")
            .attr("x1", 0)
            .attr("y1", -5)
            .attr("x2", 0)
            .attr("y2", 5)
            .style("opacity", d.hasSimilarity ? 1 : 0);  // 初始為顯示，點擊後隱藏
    });
}


