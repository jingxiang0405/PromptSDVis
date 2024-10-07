function initZoomableCirclepacking(data) {


    function renderChart(data) {
        const svg = d3.select("#zoomable-circle-packing")


        // 移除先前的圖形元素，避免疊加
        svg.selectAll("*").remove();


        const width = svg.attr("width");
        const height = svg.attr("height");

        const color = d3.scaleLinear()
            .domain([0, 5])
            .range(["white", "white"])
            .interpolate(d3.interpolateHcl);

        const pack = data => d3.pack()
            .size([width, height])
            .padding(3)(
                d3.hierarchy(data)
                    .sum(d => d.value)
                    .sort((a, b) => b.value - a.value)
            );

        const root = pack(data);

        svg.attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
            .attr("width", width)
            .attr("height", height)
            .attr("style", `max-width: 100%; height: auto; display: block; background: ${color(0)};`);

        const node = svg.append("g")
            .selectAll("circle")
            .data(root.descendants())
            .join("circle")
            // 动态检查是否有 img 属性，并根据 depth 使用图像或颜色
            .attr("fill", d => d.data.img ? `url(#pattern-${d.data.name}-${d.depth})` : "white")
            .attr("stroke", "black")  // 设置圆形边框为黑色
            .attr("stroke-width", 2)
            .attr("pointer-events", d => !d.children ? "none" : null)
            .on("mouseover", function () {
                d3.select(this).attr("stroke", "red");
            })
            .on("mouseout", function () {

                d3.select(this).attr("stroke", "black");
            })
            .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

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

        const clickedLabels = []; // 存储通过点击添加的标签名称

        const label = svg.append("g")
            .style("font", "20px sans-serif")
            .attr("text-anchor", "middle")
            .selectAll("text")
            .data(root.descendants())
            .join("text")
            .style("fill-opacity", d => d.parent === root ? 1 : 0)
            .style("display", d => d.parent === root ? "inline" : "none")
            .attr("pointer-events", d => d.depth > 1 ? "auto" : "none")
            .text(d => {
                if (d.depth === 1) {
                    return `${d.data.name}`;
                } else if (d.depth === 2) {

                    return `${d.data.name}`;
                } else if (d.depth === 3) {

                    return `${d.data.name}`;
                } else {
                    return "";  // 其他層級不顯示名稱
                }
            }).on("click", function (event, d) {
                if (d.depth > 1) {
                    event.stopPropagation(); // 阻止缩放
                    handleLabelClick(d.data.name);
                }
            });

        function handleLabelClick(labelName) {
            // 检查是否已在 clickedLabels 中
            const index = clickedLabels.lastIndexOf(labelName);
            if (index !== -1) {
                // 存在，删除最近添加的一个
                clickedLabels.splice(index, 1);
            } else {
                // 不存在，添加到 clickedLabels
                clickedLabels.push(labelName);
            }

            // 更新标签选中状态
            const isSelected = clickedLabels.includes(labelName);
            label.filter(d => d.data.name === labelName)
                .classed('selected', isSelected);

            // 更新 textarea 内容
            updateTextarea();
        }

        function updateTextarea() {
            const textareaContent = initialLabels.concat(clickedLabels);  // 合併陣列
            textarea.property('value', textareaContent.join(', '));  // 用逗號串接字串，並設置到 textarea 中
        }

        // 初始时，标签不受 textarea 内容影响，不需要设置选中状态

        svg.on("click", (event) => zoom(event, root));
        let focus = root;
        let view;

        zoomTo([focus.x, focus.y, focus.r * 2]);

        function zoomTo(v) {
            const k = width / v[2];
            view = v;

            label.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
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
                .filter(function (d) {
                    return d.parent === focus || this.style.display === "inline";
                })
                .transition(transition)
                .style("fill-opacity", d => d.parent === focus ? 1 : 0)
                .on("start", function (d) {
                    if (d.parent === focus) this.style.display = "inline";
                })
                .on("end", function (d) {
                    if (d.parent !== focus) this.style.display = "none";
                });
        }
    }
    function noData(){
        const svg = d3.select("#zoomable-circle-packing");
        svg.selectAll("*").remove();  // 清空畫布
        svg.append("text")
            .attr("x", svg.attr("width") / 2)
            .attr("y", svg.attr("height") / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "24px")
            .text("No data available for visualization");
    }
    if (data.children.length === 0){
        noData();
    } else {
        renderChart(data);
    }
   
}