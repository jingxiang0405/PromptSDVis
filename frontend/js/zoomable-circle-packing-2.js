function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // 避免正則表達式的特殊字符影響
}

function createSentenceWithSpans(sentence, prediction) {
    // 確保 prediction 是有效的對象
    if (!prediction || typeof prediction !== 'object') {
        console.error('Invalid prediction:', prediction);
        return sentence;
    }

    // 將分類項目收集到一個陣列，並按長度降序排序
    const classifiedTerms = Object.entries(prediction)
        .map(([term]) => term)
        .sort((a, b) => b.length - a.length); // 確保長詞優先處理

    if (classifiedTerms.length === 0) {
        return sentence; // 沒有分類項目，返回原句子
    }

    // 遍歷 classifiedTerms，逐一替換
    let modifiedSentence = sentence;
    classifiedTerms.forEach((term) => {
        const regex = new RegExp(`(${escapeRegExp(term)})(?![^<]*>)`, 'gi'); // 確保不是已被 <span> 包裹的內容
        modifiedSentence = modifiedSentence.replace(
            regex,
            `<span class="classified-term">$1</span>` // 使用通用的 classified-term 標籤
        );
    });

    return modifiedSentence;
}


function creatTable(data) {

    let groupedData = [];

    data.test_results.forEach(result => {
        const testPrompt = result.test_prompt;

        const simResults = result.diffusiondb_sim_search_results.map(trainItem => {
            // 解析 prediction
            let predictionObj = {};
            if (trainItem.prediction && trainItem.prediction !== '{}') {
                try {
                    // Replace single quotes with double quotes for valid JSON
                    predictionObj = JSON.parse(trainItem.prediction.replace(/'/g, '"'));
                } catch (e) {
                    console.error('Error parsing prediction:', trainItem.prediction);
                }
            }
            return {
                image: trainItem.train_idx,
                prompt: trainItem.prompt,
                prediction: predictionObj // 添加解析后的 prediction 对象
            };
        });

        // 将数据加入 groupedData 数组
        groupedData.push({
            testPrompt: testPrompt,
            items: simResults
        });
    });

    // 获取表格元素
    const table = document.querySelector("#image-sentence-table");

    // 先清空表格的內容
    table.innerHTML = "";

    // 动态创建一个新的 <tbody>
    const tbody = document.createElement("tbody");

    // 遍历 groupedData，生成表格行
    groupedData.forEach(group => {
        const testPrompt = group.testPrompt;
        const items = group.items;

        const trTestPrompt = document.createElement("tr");
        const tdTestPrompt = document.createElement("td");

        tdTestPrompt.textContent = `${testPrompt}`;
        tdTestPrompt.colSpan = 2; // 假设有两列：图片和句子 相似度 seed
        tdTestPrompt.classList.add("test-prompt-cell"); // 可选：添加类名以定制样式

        trTestPrompt.appendChild(tdTestPrompt);
        tbody.appendChild(trTestPrompt);

        // 为每个 item 创建一行，包含图片和句子
        items.forEach(item => {
            const tr = document.createElement("tr");
            tr.setAttribute('data-idx', item.image); // 添加 idx 属性 image_1539.png

            // 创建图片单元格
            const tdImage = document.createElement("td");
            const img = document.createElement("img");
            img.src = `/data/diffusiondb_data/random_5k/diffusiondb_random_5k/images/${item.image}`; // 请替换为实际的图片路径
            img.alt = item.image;
            img.width = 100;
            tdImage.appendChild(img);

            // 创建句子单元格
            const tdPrompt = document.createElement("td");

            // 在句子中标记被分类的内容
            const sentenceWithSpans = createSentenceWithSpans(item.prompt, item.prediction);
            console.log(sentenceWithSpans);
            tdPrompt.innerHTML = sentenceWithSpans; // 设置带有 <span> 标签的句子

            // 将单元格添加到行
            tr.appendChild(tdImage);
            tr.appendChild(tdPrompt);

            // 将行添加到表格主体
            tbody.appendChild(tr);
        });
    });

    // 添加新的 <tbody> 到表格中
    table.appendChild(tbody);


}



function transformTestResultsToData(data) {
    const nodeData = {
        "name": "",
        "children": []
    };

    // Helper function to find or create category node
    function findOrCreateNode(parentNode, nodeName) {
        let node = parentNode.children.find(child => child.name === nodeName);
        if (!node) {
            node = {
                "name": nodeName,
                "children": []
            };
            parentNode.children.push(node);
        }
        return node;
    }

    // Map to keep track of contents under each category to avoid duplicates
    const categoryContentMap = {};

    // Iterate over test_results
    data.test_results.forEach(testResult => {

        // Process test_prediction
        if (testResult.test_prediction && Object.keys(testResult.test_prediction).length > 0) {
            for (let content in testResult.test_prediction) {
                let category = testResult.test_prediction[content];

                // Find or create category node
                let categoryNode = findOrCreateNode(nodeData, category);

                // Initialize the content set for this category if not exist
                if (!categoryContentMap[category]) {
                    categoryContentMap[category] = new Set();
                }

                // Add the content if it's not already added
                if (!categoryContentMap[category].has(content)) {
                    categoryContentMap[category].add(content);
                    categoryNode.children.push({
                        "name": content,
                        "value": 100,
                        "isTestPrediction": true // 标记为 test_prediction

                    });
                }
            }
        }

        // Process top3_similar_train
        testResult.diffusiondb_sim_search_results.forEach(trainItem => {
            let prediction = trainItem.prediction;
            if (prediction && prediction !== '{}') {
                // Parse the prediction string into an object
                let predictionObj;
                try {
                    // Replace single quotes with double quotes for valid JSON
                    predictionObj = JSON.parse(prediction.replace(/'/g, '"'));
                } catch (e) {
                    console.error('Error parsing prediction:', prediction);
                    return;
                }

                // Iterate over the predictions
                for (let content in predictionObj) {
                    let category = predictionObj[content];

                    // Find or create category node
                    let categoryNode = findOrCreateNode(nodeData, category);

                    // Initialize the content set for this category if not exist
                    if (!categoryContentMap[category]) {
                        categoryContentMap[category] = new Set();
                    }

                    // Add the content if it's not already added
                    if (!categoryContentMap[category].has(content)) {
                        categoryContentMap[category].add(content);
                        categoryNode.children.push({
                            "name": content,
                            "value": 100,
                            "isTestPrediction": false, // 标记为 train_prediction
                            "idx": trainItem.train_idx // 添加 idx 属性
                        });
                    }
                }
            }
        });
    });

    return nodeData;
}

function renderChart(_resultData) {
    const svg = d3.select("#zoomable-circle-packing")


    // 移除先前的圖形元素，避免疊加
    svg.selectAll("*").remove();


    const width = svg.attr("width");
    const height = svg.attr("height");

    const color = d3.scaleLinear()
        .domain([0, 5])
        .range(["white", "white"])
        .interpolate(d3.interpolateHcl);

    const pack = _resultData => d3.pack()
        .size([width, height])
        .padding(3)(
            d3.hierarchy(_resultData)
                .sum(d => d.value)
                .sort((a, b) => b.value - a.value)
        );

    const root = pack(_resultData);

    svg.attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
        .attr("width", width)
        .attr("height", height)
        .attr("style", `max-width: 100%; height: auto; display: block; background: ${color(0)};`);

    const node = svg.append("g")
        .selectAll("circle")
        .data(root.descendants())
        .join("circle")
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("pointer-events", d => !d.children ? "none" : null)
        .on("mouseover", function () {
            d3.select(this).attr("stroke", "red");
        })
        .on("mouseout", function () {
            d3.select(this).attr("stroke", "black");
        })
        .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

    const clickedLabels = []; // 存储通过点击添加的标签名称

    function updateNodeColors() {
        node.attr("fill", d => {
            if (d.depth === 2) {
                if (clickedLabels.includes(d.data.name)) {
                    return "#f8db64"; // 被选中的节点设为橘色
                } else if (d.data.isTestPrediction) {
                    return "#FFFACD"; // 淡黄色
                } else {
                    return "#ADD8E6"; // 默认颜色（淡蓝色）
                }
            } else {
                return "white"; // 默认颜色
            }
        });
    }

    // 初始化节点的填充颜色
    updateNodeColors();

    const defs = svg.append("defs");

    root.descendants().filter(d => d.data.img).forEach(d => {
        defs.append("pattern")
            .attr("id", `pattern-${d.data.name}-${d.depth}`)
            .attr("patternContentUnits", "objectBoundingBox")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 1)
            .attr("height", 1)
            .append("image")
            .attr("xlink:href", d.data.img)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .attr("width", 1)
            .attr("height", 1);
    });

    // 新增部分
    const textarea = d3.select("#prompt");

    const initialText = textarea.property('value');
    const initialLabels = initialText ? initialText.split('\n').map(s => s.trim()).filter(s => s) : [];



    const label = svg.append("g")
        .selectAll("g")
        .data(root.descendants())
        .join("g")
        .attr("class", "label-group")
        .style("fill-opacity", d => d.parent === root ? 1 : 0)
        .style("display", d => d.parent === root ? "inline" : "none")
        .attr("pointer-events", d => d.depth > 1 ? "auto" : "none")
        .on("click", function (event, d) {
            if (d.depth > 1) {
                event.stopPropagation(); // 阻止缩放
                handleLabelClick(d.data.name); // 传递标签文字内容
            }
        });

    label.each(function (d) {
        const g = d3.select(this);

        // 添加文字元素
        const text = g.append("text")
            .text(d.data.name)
            .attr("text-anchor", "middle")
            .attr("data-idx", d.data.idx) // 添加 idx 属性
            .style("stroke", "white")
            .style("stroke-width", "2px")
            .style("paint-order", "stroke")
            .each(function () {
                if (d.depth === 1 || d.depth === 2) {
                    fitTextToCircle(d3.select(this), d.r);
                }
            });

        // 如果是第一层，添加背景矩形
        if (d.depth === 1) {
            // 在获取边界框之前，需要确保文字已经渲染
            // 可以使用 setTimeout 或者 requestAnimationFrame，或者在下一次事件循环中执行
            setTimeout(() => {
                const bbox = text.node().getBBox();

                // 在文字之前插入矩形
                g.insert("rect", "text")
                    .attr("x", bbox.x - 5) // 留出一些边距
                    .attr("y", bbox.y - 2)
                    .attr("width", bbox.width + 10)
                    .attr("height", bbox.height + 4)
                    .attr("fill", "white");
            }, 0);
        }
    });


    // fitTextToCircle 函數，調整文字大小以適應圓圈
    function fitTextToCircle(textElement, radius) {
        const maxWidth = radius * 1.8;  // 圓圈內的最大文字寬度
        let fontSize = 20;  // 初始字體大小
        const minFontSize = 10;  // 設定最小字體大小
        let maxIterations = 4;  // 最大迭代次數以防止無限循環
        textElement.style("font-size", `${fontSize}px`);
        // 將原始文本內容拆分成單詞
        let words = textElement.text().split(" ");
        let lines = [];
        // 如果文本包含多個單詞，先嘗試換行
        if (words.length > 1) {
            let line = "";
            words.forEach(word => {
                let testLine = line + word + " ";
                textElement.text(testLine.trim());
                if (textElement.node().getBBox().width > maxWidth && line !== "") {
                    lines.push(line.trim());
                    line = word + " ";
                } else {
                    line = testLine;
                }
            });
            lines.push(line.trim());

            // 清空並設置換行後的文本
            textElement.text("");
            lines.forEach((line, i) => {
                textElement.append("tspan")
                    .text(line)
                    .attr("x", 0)
                    .attr("dy", i === 0 ? 0 : "1.2em");  // 設置行間距
            });
        }

        // 確保文本在圓圈內，並逐步縮小字體
        while (true) {
            const bbox = textElement.node().getBBox();
            if ((bbox.width <= maxWidth && bbox.height <= radius * 1.6) || fontSize <= minFontSize || maxIterations-- <= 0) {
                break;
            }

            fontSize = Math.max(minFontSize, fontSize - 1);
            textElement.style("font-size", `${fontSize}px`);
        }

        // 如果有多行，則調整行的位置以保持居中
        const lineHeight = textElement.selectAll("tspan").nodes().length * fontSize * 1.2;  // 計算多行文本高度
        textElement.attr("y", -lineHeight / 2);  // 確保文本的行高度整體居中

        // 如果是單行文本，也進行垂直居中調整
        if (lines.length <= 1) {
            const textHeight = textElement.node().getBBox().height;
            textElement.attr("dy", textHeight / 4);
        }
    }


    function highlightTableText() {
        // 先移除之前的高亮
        d3.selectAll("#image-sentence-table .classified-term").classed('highlight', false);

        // 对于每个选中的标签，查找并高亮对应的表格文字
        clickedLabels.forEach(labelName => {
            d3.selectAll("#image-sentence-table .classified-term")
                .filter(function () {
                    return this.textContent.trim() === labelName;
                })
                .classed('highlight', true);
        });
    }

    function handleLabelClick(labelName) {
        const index = clickedLabels.indexOf(labelName);
        if (index !== -1) {
            clickedLabels.splice(index, 1);
        } else {
            clickedLabels.push(labelName);
        }

        const isSelected = clickedLabels.includes(labelName);
        label.filter(d => d.data.name === labelName)
            .classed('selected', isSelected);

        updateNodeColors();
        
        updateTextarea();
        highlightTableText();
    }

    function updateTextarea() {
        const textareaContent = initialLabels.concat(clickedLabels);  // 合併陣列
        textarea.property('value', textareaContent.join(', '));  // 用逗號串接字串，並設置到 textarea 中
    }

    document.querySelectorAll('.classified-term').forEach(termElement => {
        termElement.addEventListener('click', (event) => {
            const term = event.target.textContent.trim();
            handleLabelClick(term); // 传递点击的 term
        });
    });
    // 初始时，标签不受 textarea 内容影响，不需要设置选中状态

    svg.on("click", (event) => zoom(event, root));
    let focus = root;
    let view;

    zoomTo([focus.x, focus.y, focus.r * 2]);

    function zoomTo(v) {
        const k = width / v[2];
        view = v;

        label.attr("transform", d => {
            const x = (d.x - v[0]) * k;
            const y = (d.y - v[1]) * k;
            if (d.depth === 1) {
                // 将深度为1的标签放置在圆圈的上方
                return `translate(${x},${y - d.r * k * 0.8})`;
            } else {
                // 将深度为2的标签放置在圆圈的中心
                return `translate(${x},${y})`;
            }
        });

        node.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
        node.attr("r", d => d.r * k);
    }

    function zoom(event, d) {
        const focus0 = focus;
        focus = d;

        const transition = svg.transition()
            .duration(750)
            .tween("zoom", d => {
                const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
                return t => zoomTo(i(t));
            });

        label
            .transition(transition)
            .style("fill-opacity", d => (d.parent === focus || d === focus) ? 1 : 0)
            .on("start", function (d) {
                if (d.parent === focus || d === focus) this.style.display = "inline";
            })
            .on("end", function (d) {
                if (d.parent !== focus && d !== focus) this.style.display = "none";
            });
    }

}

function initZoomableCirclepacking(data) {
    // 假设您的数据已存储在 testResults 中
    creatTable(data);

    // Run the test and display result
    const resultData = transformTestResultsToData(data);

    renderChart(resultData);

}







