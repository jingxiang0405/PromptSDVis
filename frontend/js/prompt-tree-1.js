function initRelationshipGraph(data, inputText) {
    const boxHeight = 20;

    // 调整节点间距
    const gap = { width: 50, height: 12 };
    //const margin = { top: 16, right: 16, bottom: 16, left: 16 };
    let Nodes = [], links = [], lvlCount = 0;

    // 创建 SVG 画布
    const svg = d3.select("#prompt-tree-svg");
    const width = +svg.attr("width"); // 转换为数字
    const height = +svg.attr("height");
    svg.selectAll("*").remove();
    // 初始化高亮节点集合
    let highlightedNodes = new Set();

    // 创建一个包含所有图形元素的组，以便缩放和平移
    const container = svg.append("g");

    // 定义缩放行为
    const zoom = d3.zoom()
        .scaleExtent([0.5, 5]) // 缩放比例范围
        .on("zoom", (event) => {
            container.attr("transform", event.transform);
        });

    // 应用缩放行为到 SVG
    svg.call(zoom);

    // 在 container 背景上添加透明矩形，监听点击事件
    container.append("rect")
        .attr("width", width)
        .attr("height", height)
        .style("fill", "none")
        .style("pointer-events", "all")
        .on("click", function () {
            clearSelection();
        });

    // 对角线函数
    const diagonal = d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x);

    // 根据 lvl 和 name 查找节点的函数
    function find(lvl, name) {
        return Nodes.find((node) => node.lvl === lvl && node.name === name) || null;
    }


    // 函数用于打印所有可能的句子组合
    function printCombinations(highlightedNodes, baseSentence) {
        // 将节点按层级排序，以便生成正确顺序的句子
        let sortedNodes = Array.from(highlightedNodes).sort((a, b) => a.lvl - b.lvl);

        // 使用正则表达式捕获括号中的内容，保留原始分隔符
        let parts = [];
        let regex = /\([^)]*\)|[^()]+/g;
        let match;
        while ((match = regex.exec(baseSentence)) !== null) {
            parts.push(match[0]);
        }

        // 创建所有的替换组合
        let combinations = [];

        // 递归函数来生成所有可能的句子组合
        function generateCombinations(index, currentCombination) {
            if (index === parts.length) {
                combinations.push(currentCombination.join(''));
                return;
            }

            let part = parts[index];
            if (part.startsWith('(')) {
                // 处理括号内的多选部分
                let options = part.slice(1, -1).split(",").map(option => option.trim());
                options.forEach(option => {
                    if (sortedNodes.some(node => node.name === option)) {
                        generateCombinations(index + 1, [...currentCombination, option]);
                    }
                });
            } else {
                // 处理非括号部分，直接添加
                generateCombinations(index + 1, [...currentCombination, part]);
            }
        }

        // 生成所有可能的组合
        generateCombinations(0, []);



        const combinationSet = new Set(combinations);
        // 2. 收集所有需要更新的 rect
        const rectUpdates = [];

        // 選擇所有包含圖像的 <div>
        d3.selectAll("#scatter image").each(function () {
            const imgElement = d3.select(this);
            const imgTitle = imgElement.attr("data-title")?.trim();

            console.log('imgTitle: ' + imgTitle);

            // 初始化 matched 標誌，檢查是否匹配 combinationSet
            const matched = combinationSet.has(imgTitle);
            console.log('qqqq')
            console.log(matched)
            // 找到對應的 <rect>
            const rectElement = d3.select(`#rect-${imgElement.attr("id").replace('.png', '')}`);
            console.log('sssss')
            console.log(`#rect-${imgElement.attr("id").replace('.png', '')}`)
            // 儲存原始外框顏色到屬性中
            if (!rectElement.attr("data-original-stroke")) {
                rectElement.attr("data-original-stroke", rectElement.style("stroke") || "black");
            }

            // 收集需要更新的 rect 信息
            rectUpdates.push({
                element: rectElement,
                stroke: matched ? "orange" : rectElement.attr("data-original-stroke"),
                strokeWidth: matched ? "3" : "2"
            });
        });

        // 3. 統一應用樣式更新
        rectUpdates.forEach(({ element, stroke, strokeWidth }) => {
            element.style("stroke", stroke).style("stroke-width", strokeWidth);
        });

    }
    // 清除选中状态的函数
    function clearSelection() {

        d3.selectAll("#scatter image").each(function () {
            const imgElement = d3.select(this);
            imgId = imgElement.attr("id").replace('.png', '')
            if (!imgId) return; // 確保 id 存在

            // 找到對應的 <rect>，恢復原始外框顏色
            const rectElement = d3.select(`#rect-${imgId}`);
            console.log('----------')
            console.log(`#rect-${imgElement.attr("id").replace('.png', '')}`)
            const originalStroke = rectElement.attr("data-original-stroke");

            if (originalStroke) {
                rectElement
                    .style("stroke", originalStroke)
                    .style("stroke-width", "2");
            }
        });
        // 清除高亮狀態
        highlightedNodes.clear();
        update_selection();
    }
    // 更新选中状态的函数
    function update_selection(inputText) {
        console.log('update_selection:' + inputText)
        // 清除所有节点和链接的样式
        container.selectAll(".unit rect")
            .style("fill", "#CCC")
            .style("stroke", null);

        container.selectAll(".link")
            .style("stroke", "#ccc")
            .style("stroke-width", "2.5px");

        // 如果没有任何节点被高亮，则不进行任何操作
        if (highlightedNodes.size === 0) {
            return;
        }

        // 记录每个层级已高亮的节点
        let highlightedNodesPerLevel = {};

        // 构建已高亮节点的层级映射
        highlightedNodes.forEach(node => {
            if (!highlightedNodesPerLevel[node.lvl]) {
                highlightedNodesPerLevel[node.lvl] = [];
            }
            highlightedNodesPerLevel[node.lvl].push(node);
        });

        // 对于每个层级，如果没有节点被高亮，则高亮 order = 0 的节点
        let levels = Array.from(new Set(Nodes.map(n => n.lvl))).sort((a, b) => a - b);

        levels.forEach(lvl => {
            if (!highlightedNodesPerLevel[lvl] || highlightedNodesPerLevel[lvl].length === 0) {
                // 该层级没有节点被高亮，找到 order = 0 的节点并高亮
                let node = Nodes.find(n => n.lvl === lvl && n.order === 0);
                if (node) {
                    highlightedNodes.add(node);
                    // 更新高亮节点的层级映射
                    if (!highlightedNodesPerLevel[lvl]) {
                        highlightedNodesPerLevel[lvl] = [];
                    }
                    highlightedNodesPerLevel[lvl].push(node);
                }
            }
        });

        // 应用高亮样式到已高亮的节点
        highlightedNodes.forEach(node => {
            container.select("#" + node.id)
                .style("fill", "orange")
                .style("stroke", "orange");
        });

        // 收集已高亮节点的 ID
        let highlightedNodeIds = new Set(Array.from(highlightedNodes).map(n => n.id));

        // 高亮连接已高亮节点之间的链接
        links.forEach(link => {
            if (highlightedNodeIds.has(link.source.id) && highlightedNodeIds.has(link.target.id)) {
                container.select("#" + link.id)
                    .style("stroke", "orange")
                    .style("stroke-width", "2.5px");
            }
        });

        // 打印所有可能的句子组合
        printCombinations(highlightedNodes, inputText);
    }

    // 渲染图表
    function renderGraph(data) {
        let count = [];
        let maxNodeWidthPerLevel = []; // 记录每个层级的最大节点宽度

        // 计算层级数量
        data.Nodes.forEach(function (d) {
            count[d.lvl] = count[d.lvl] || 0;
        });
        lvlCount = count.length;

        // 创建用于测量文本宽度的临时 SVG
        const tempSVG = d3.select("body").append("svg").attr("class", "tempSVG").style("visibility", "hidden");

        // 测量每个节点名称的文本宽度
        data.Nodes.forEach(function (d, i) {
            // 创建临时文本元素
            const tempText = tempSVG.append("text")
                .attr("class", "tempText")
                .style("font-family", "sans-serif")
                .style("font-size", "12px")
                .text(d.name);

            // 获取文本宽度
            const textWidth = tempText.node().getBBox().width;

            // 移除临时文本元素
            tempText.remove();

            // 为文本添加左右内边距
            const boxWidth = textWidth + 20; // 左右各 10 像素内边距

            d.boxWidth = boxWidth; // 将节点宽度存储在节点数据中

            // 更新每个层级的最大节点宽度
            if (!maxNodeWidthPerLevel[d.lvl] || boxWidth > maxNodeWidthPerLevel[d.lvl]) {
                maxNodeWidthPerLevel[d.lvl] = boxWidth;
            }
        });

        // 移除临时 SVG
        tempSVG.remove();

        // 计算图形的总宽度
        let graphWidth = 0;
        for (let l = 0; l < lvlCount; l++) {
            graphWidth += (maxNodeWidthPerLevel[l] || 0) + (l > 0 ? gap.width : 0);
        }

        // 计算图形的总高度
        let levelHeights = [];
        for (let l = 0; l < lvlCount; l++) {
            let nodesInLevel = data.Nodes.filter(d => d.lvl === l);
            let levelHeight = nodesInLevel.length * (boxHeight + gap.height) - gap.height; // 减去最后一个 gap.height
            levelHeights.push(levelHeight > 0 ? levelHeight : 0);
        }
        let graphHeight = Math.max(...levelHeights);

        // 计算偏移量，使图形居中
        let offsetX = (width - graphWidth) / 2;
        let offsetY = (height - graphHeight) / 2;

        // 设置每个节点的坐标，并添加 `order` 属性
        data.Nodes.forEach(function (d, i) {
            if (!count[d.lvl]) count[d.lvl] = 0;
            // 计算 x 坐标，基于前面层级的最大节点宽度
            let xOffset = 0;
            for (let l = 0; l < d.lvl; l++) {
                xOffset += (maxNodeWidthPerLevel[l] || 0) + gap.width;
            }
            d.x = xOffset + offsetX;
            d.y = (boxHeight + gap.height) * count[d.lvl] + offsetY;
            d.id = "n_" + d.lvl + "_" + d.name.replace(/\s+/g, "_").replace(/[^\w]/g, '');
            d.order = count[d.lvl]; // 添加 order 属性
            count[d.lvl] += 1;
            Nodes.push(d);
        });

        // 建立连结
        data.links.forEach(function (d) {
            const sourceNode = find(d.source_lvl, d.source);
            const targetNode = find(d.target_lvl, d.target);
            if (sourceNode && targetNode) {
                links.push({
                    source: sourceNode,
                    target: targetNode,
                    id: "l_" + sourceNode.id + "_" + targetNode.id,
                });
            }
        });

        // 在 container 中绘制节点
        const node = container
            .selectAll(".unit")
            .data(Nodes)
            .join("g")
            .attr("class", "unit");

        node
            .append("rect")
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("id", (d) => d.id)
            .attr("width", (d) => d.boxWidth)
            .attr("height", boxHeight)
            .attr("rx", 6)
            .attr("ry", 6)
            .style("fill", "#CCC")
            .style("cursor", "pointer")
            .on("click", function (event, d) {
                // 切换节点的高亮状态
                if (highlightedNodes.has(d)) {
                    highlightedNodes.delete(d);
                } else {
                    highlightedNodes.add(d);
                }
                update_selection(inputText);
                event.stopPropagation();
            });

        node
            .append("text")
            .attr("x", (d) => d.x + 10) // 左内边距
            .attr("y", (d) => d.y + 15)
            .style("fill", "black")
            .style("font-family", "sans-serif")
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .text((d) => d.name);

        // 在 container 中绘制连结
        container.selectAll(".link")
            .data(links)
            .join("path")
            .attr("class", "link")
            .attr("id", (d) => d.id)
            .attr("d", function (d) {
                const sourcePoint = { x: d.source.y + boxHeight / 2, y: d.source.x + d.source.boxWidth };
                const targetPoint = { x: d.target.y + boxHeight / 2, y: d.target.x };
                return diagonal({ source: sourcePoint, target: targetPoint });
            })
            .style("fill", "none")
            .style("stroke", "#ccc")
            .style("stroke-width", "2.5px");
    }

    // 调用渲染图表的函数
    renderGraph(data);
}
