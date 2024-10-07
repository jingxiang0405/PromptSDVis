function initZoomableCirclepacking(data) {
    // 假數據，新增 img 屬性
    /* 
    const fakeData = {
        "name": "flare",
        "children": [
            {
                "name": "Subject term",
                "children": [
                    { "name": "CommunityStructure", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" },
                    { "name": "AgglomerativeCluster", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            },
            {
                "name": "Style modifier",
                "children": [
                    { "name": "Easing", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            },
            {
                "name": "Style modifier",
                "children": [
                    { "name": "Easing", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            }
            ,
            {
                "name": "Style modifier",
                "children": [
                    { "name": "Easing", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            },
            {
                "name": "Style modifier",
                "children": [
                    { "name": "Easing", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            },
            {
                "name": "Style modifier",
                "children": [
                    { "name": "Easing", "value": 600, "img": "https://i.postimg.cc/NjnNrbqv/temp-Imageqh-A8-Eh.avif" }
                ]
            }
        ]
    };
*/
    function renderChart(data) {
        const svg = d3.select("#zoomable-circle-packing")
        const width = svg.attr("width");
        const height = svg.attr("height");

        const color = d3.scaleLinear()
            .domain([0, 5])
            .range(["hsl(152,80%,80%)", "hsl(228,30%,40%)"])
            .interpolate(d3.interpolateHcl);

        const pack = data => d3.pack()
            .size([width, height])
            .padding(3)(
                d3.hierarchy(data)
                    .sum(d => d.value)
                    .sort((a, b) => b.value - a.value)
            );

        // 移除現有的 <g> 元素
        svg.selectAll("g").remove();

        const root = pack(data);

        svg.attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
            .attr("width", width)
            .attr("height", height)
            .attr("style", `max-width: 100%; height: auto; display: block; background: ${color(0)};`);

        const node = svg.append("g")
            .selectAll("circle")
            .data(root.descendants().slice(1))
            .join("circle")
            .attr("fill", d => d.data.img ? `url(#pattern-${d.data.name})` : color(d.depth))
            .attr("pointer-events", d => !d.children ? "none" : null)
            .on("mouseover", function () {
                d3.select(this).attr("stroke", "#000");
            })
            .on("mouseout", function () {
                d3.select(this).attr("stroke", null);
            })
            .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

        const defs = svg.append("defs");
        
        root.descendants().filter(d => d.data.img).forEach(function (d) {
            defs.append("pattern")
                .attr("id", `pattern-${d.data.name}`)
                .attr("patternUnits", "userSpaceOnUse")
                .attr("width", d.r * 2)
                .attr("height", d.r * 2)
                .attr("x", -d.r)
                .attr("y", -d.r)
                .append("image")
                .attr("x", "0%")
                .attr("y", "0%")
                .attr("width", d.r * 2)
                .attr("height", d.r * 2)
                .attr("href", d.data.img);
        });

        // 新增部分
        const textarea = d3.select("#prompt");

        const initialText = textarea.property('value');
        const initialLabels = initialText ? initialText.split('\n').map(s => s.trim()).filter(s => s) : [];

        const clickedLabels = []; // 存储通过点击添加的标签名称

        const label = svg.append("g")
            .style("font", "10px sans-serif")
            .attr("text-anchor", "middle")
            .selectAll("text")
            .data(root.descendants())
            .join("text")
            .attr("pointer-events", d => d.depth > 1 ? "auto" : "none")
            .style("fill-opacity", d => d.parent === root ? 1 : 0)
            .style("display", d => d.parent === root ? "inline" : "none")
            .text(d => d.data.name)
            .on("click", function (event, d) {
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
            const textareaContent = initialLabels.concat(clickedLabels);
            textarea.property('value', textareaContent.join('\n'));
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

            root.descendants().filter(d => d.data.img).forEach(d => {
                const pattern = d3.select(`#pattern-${d.data.name}`);
                const newRadius = d.r * k;

                pattern
                    .attr("width", newRadius * 2)
                    .attr("height", newRadius * 2)
                    .attr("x", -newRadius)
                    .attr("y", -newRadius);

                pattern.select("image")
                    .attr("width", newRadius * 2)
                    .attr("height", newRadius * 2);
            });
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

    renderChart(data);
}