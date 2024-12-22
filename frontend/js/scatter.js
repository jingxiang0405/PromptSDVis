function initScatterPlot(images) {
    // Tooltip
    const tooltip = d3.select('#tooltip');

    // Main Map
    const svg = d3.select("#scatter");
    const width = +svg.attr("width");
    const height = +svg.attr("height");

    // Mini Map
    const gElement = d3.select("#scatter-minimap");
    const rectElement = gElement.select("rect");

    // 取得 mini map 尺寸
    const miniWidth = +rectElement.attr("width");
    const miniHeight = +rectElement.attr("height");

    const xMin = d3.min(images, d => d.x);
    const xMax = d3.max(images, d => d.x);
    const yMin = d3.min(images, d => d.y);
    const yMax = d3.max(images, d => d.y);

    // 在原始的資料範圍上加上一些邊距比例 (例如 5%)
    const xPadding = (xMax - xMin) * 0.05;
    const yPadding = (yMax - yMin) * 0.05;

    const xDomain = [xMin - xPadding, xMax + xPadding];
    const yDomain = [yMin - yPadding, yMax + yPadding];

    const xScaleMain = d3.scaleLinear()
        .domain(xDomain)
        .range([0, width]);

    const yScaleMain = d3.scaleLinear()
        .domain(yDomain)
        .range([0, height]);

    // Mini map 使用相同 domain 對應 [0, miniWidth], [0, miniHeight]
    const xScaleMini = d3.scaleLinear()
        .domain(xDomain)
        .range([0, miniWidth]);

    const yScaleMini = d3.scaleLinear()
        .domain(yDomain)
        .range([0, miniHeight]);

    // 清空 mini map 的內容，但保留標籤
    gElement.selectAll("g.mini-group").remove();
    const miniG = gElement.append("g").attr("class", "mini-group");

    // 在 mini map 上繪製資料點
    miniG.selectAll("circle")
        .data(images)
        .enter()
        .append("circle")
        .attr('cx', d => xScaleMini(d.x))
        .attr('cy', d => yScaleMini(d.y))
        .attr('r', 2)
        .attr('fill', 'blue');

    // 刪除舊的 viewport
    gElement.select("rect.viewport").remove();

    // 新增 viewport
    const viewport = gElement.append("rect")
        .attr("class", "viewport")
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-width", 1);

    // 移除主地圖上舊的 g
    svg.selectAll("g.main-group").remove();
    const g = svg.append('g').attr("class", "main-group");

    // 將資料座標映射到主地圖範圍內
    // 初始位置已縮放到主地圖大小內
    const imageElements = g.selectAll('image')
        .data(images)
        .enter()
        .append('image')
        .attr('id', d => d.id)
        .attr('data-title', d => d.title)
        .attr('xlink:href', d => d.src)
        .attr('x', d => xScaleMain(d.x))
        .attr('y', d => yScaleMain(d.y))
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .on('mouseover', function (event, d) {
            tooltip.style('display', 'block')
                .html(`<img src="${d.src}" width="100" height="100"><br><strong>prompt: ${d.title}</strong><br><strong>randomseed: ${d.randomseed}</strong>`)
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY + 10}px`);
        })
        .on('mouseout', function () {
            tooltip.style('display', 'none');
        });

    imageElements.exit().remove();

    // 繪製邊框 rect（圖片邊界）
    let rects = g.selectAll('rect')
        .data(images)
        .enter()
        .append('rect')
        .attr('data-original-stroke', 'black')
        .attr('class', 'image-border')
        .attr('id', d => `rect-${d.id.replace('.png', '')}`)
        .attr('x', d => xScaleMain(d.x))
        .attr('y', d => yScaleMain(d.y))
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .style("fill", "none")
        .style("stroke", "black")
        .style("stroke-width", 2);

    rects.exit().remove();

    let currentTransform = d3.zoomIdentity;

    // 定義 zoom 行為
    let zoom = d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([1, 8])
        .on("zoom", function (event) {
            currentTransform = event.transform;

            // 更新主地圖中圖片與 rect 的位置與大小
            g.selectAll('image')
                .attr('x', d => currentTransform.applyX(xScaleMain(d.x)))
                .attr('y', d => currentTransform.applyY(yScaleMain(d.y)))
                .attr('width', d => d.width * currentTransform.k)
                .attr('height', d => d.height * currentTransform.k);

            g.selectAll('.image-border')
                .attr('x', d => currentTransform.applyX(xScaleMain(d.x)))
                .attr('y', d => currentTransform.applyY(yScaleMain(d.y)))
                .attr('width', d => d.width * currentTransform.k)
                .attr('height', d => d.height * currentTransform.k);

            // 計算主地圖目前可視範圍(以 main map 座標計算)
            const visibleX_main = -currentTransform.x / currentTransform.k;
            const visibleY_main = -currentTransform.y / currentTransform.k;
            const visibleW_main = width / currentTransform.k;
            const visibleH_main = height / currentTransform.k;

            // 將可視範圍從 main map 座標轉回資料 domain
            const domainX0 = xScaleMain.invert(visibleX_main);
            const domainY0 = yScaleMain.invert(visibleY_main);
            const domainX1 = xScaleMain.invert(visibleX_main + visibleW_main);
            const domainY1 = yScaleMain.invert(visibleY_main + visibleH_main);

            // 再將 domain 範圍映射到 mini map
            const viewportX = xScaleMini(domainX0);
            const viewportY = yScaleMini(domainY0);
            const viewportW = xScaleMini(domainX1) - xScaleMini(domainX0);
            const viewportH = yScaleMini(domainY1) - yScaleMini(domainY0);

            // 對 viewport 進行 clamp，確保不超出 mini map 範圍
            const clampedViewportX = Math.max(0, Math.min(viewportX, miniWidth - viewportW));
            const clampedViewportY = Math.max(0, Math.min(viewportY, miniHeight - viewportH));

            viewport
                .attr("x", clampedViewportX)
                .attr("y", clampedViewportY)
                .attr("width", viewportW)
                .attr("height", viewportH);
        });

    // 定義在 mini map 上的拖曳行為，使 viewport 拖曳可控制主地圖
    const drag = d3.drag()
        .on("start", function () {
            // 禁用主地圖縮放事件
            svg.on(".zoom", null);
        })
        .on("drag", function (event) {
            // 反推到 domain
            const domainX = xScaleMini.invert(event.x);
            const domainY = yScaleMini.invert(event.y);

            // 計算新的 transform，保證 viewport 不跑出範圍
            const scale = currentTransform.k;
            const newX = -xScaleMain(domainX) * scale;
            const newY = -yScaleMain(domainY) * scale;

            const transform = d3.zoomIdentity
                .translate(newX, newY)
                .scale(scale);

            svg.call(zoom.transform, transform);
        })
        .on("end", function () {
            // 恢復主地圖縮放
            svg.call(zoom);
        });

    viewport.call(drag);
    svg.call(zoom);

    // Brush
    const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('start brush', event => {
            if (event.selection) {
                const [[x0, y0], [x1, y1]] = event.selection;
                rects.each(function (d) {
                    const dx = currentTransform.applyX(xScaleMain(d.x));
                    const dy = currentTransform.applyY(yScaleMain(d.y));
                    const dw = currentTransform.k * d.width;
                    const dh = currentTransform.k * d.height;

                    const overlaps =
                        (
                            ((dx <= x0 && x0 <= (dx + dw)) ||
                                (dx <= x1 && x1 <= (dx + dw)))
                            &&
                            ((dy <= y0 && y0 <= (dy + dh)) ||
                                (dy <= y1 && y1 <= (dy + dh)))
                        ) ||
                        (
                            (x0 <= dx && x1 >= (dx + dw)) &&
                            (y0 <= dy && y1 >= (dy + dh))
                        ) ||
                        (
                            ((dx <= x0 && x0 <= (dx + dw)) ||
                                (dx <= x1 && x1 <= (dx + dw)))
                            &&
                            (y0 <= dy && y1 >= (dy + dh))
                        ) ||
                        (
                            ((dy <= y0 && y0 <= (dy + dh)) ||
                                (dy <= y1 && y1 <= (dy + dh)))
                            &&
                            (x0 <= dx && x1 >= (dx + dw))
                        );

                    d3.select(this)
                        .style("stroke", overlaps ? "red" : "black")
                        .attr('data-original-stroke', overlaps ? "red" : "black");
                });
            }
        })
        .on('end', event => {
            if (!event.selection) {
                rects.style("stroke", "black");
                rects.attr('data-original-stroke', "black");
            } else {
                const [[x0, y0], [x1, y1]] = event.selection;
                const selectedImages = [];
                rects.each(function (d) {
                    const dx = currentTransform.applyX(xScaleMain(d.x));
                    const dy = currentTransform.applyY(yScaleMain(d.y));
                    const dw = currentTransform.k * d.width;
                    const dh = currentTransform.k * d.height;

                    const overlaps =
                        (
                            ((dx <= x0 && x0 <= (dx + dw)) ||
                                (dx <= x1 && x1 <= (dx + dw)))
                            &&
                            ((dy <= y0 && y0 <= (dy + dh)) ||
                                (dy <= y1 && y1 <= (dy + dh)))
                        ) ||
                        (
                            (x0 <= dx && x1 >= (dx + dw)) &&
                            (y0 <= dy && y1 >= (dy + dh))
                        ) ||
                        (
                            ((dx <= x0 && x0 <= (dx + dw)) ||
                                (dx <= x1 && x1 <= (dx + dw)))
                            &&
                            (y0 <= dy && y1 >= (dy + dh))
                        ) ||
                        (
                            ((dy <= y0 && y0 <= (dh)) ||
                                (dy <= y1 && y1 <= (dh)))
                            &&
                            (x0 <= dx && x1 >= (dx + dw))
                        );

                    if (overlaps) {
                        selectedImages.push({ id: d.id, title: d.title, image_src: d.src });
                    }
                });
                const scatterSelectionEndEvent = new CustomEvent("scatterSelectionEnd", {
                    detail: selectedImages,
                });
                document.dispatchEvent(scatterSelectionEndEvent);
            }
        });

    const toggleBrushSwitch = document.getElementById("toggle-brush");
    toggleBrushSwitch.addEventListener("change", function () {
        if (this.checked) {
            d3.selectAll(".brush").remove();
            const brushGroup = svg.append("g").attr("class", "brush");
            brushGroup.call(brush);
            svg.on(".zoom", null);
            console.log("Brush enabled");
        } else {
            d3.selectAll(".brush").remove();
            d3.selectAll(".brush").on(".brush", null);
            rects.style("stroke", "black").attr('data-original-stroke', "black");
            svg.call(zoom);
            console.log("Brush disabled");
        }
    });
}
