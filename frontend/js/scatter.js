function initScatterPlot(images) {
    const imageOverview = d3.select('#tooltip');
    const svg = d3.select("#svg1")
    const width = svg.attr("width");
    const height = svg.attr("height");
    // 尚未解決完問題 start
    // 移除现有的 <g> 元素
    svg.selectAll("g").remove();
    const g = svg.append('g');

    // 计算 t-SNE 数据的边界框
    let minX = d3.min(images, d => d.x);
    let maxX = d3.max(images, d => d.x);
    let minY = d3.min(images, d => d.y);
    let maxY = d3.max(images, d => d.y);

    // 计算缩放比例和平移量
    let scaleX = width / (maxX - minX);
    let scaleY = height / (maxY - minY);
    let scale = Math.min(scaleX, scaleY) * 0.8; // 乘以 0.8 以留出一些边界空间

    // 设置缩放限制
    const minScale = 0.1; // 最小缩放限制
    const maxScale = 2; // 最大缩放限制

    // 调整缩放比例，保证它在合理的范围内
    scale = Math.max(minScale, Math.min(maxScale, scale));

    // 调整平移量以保证视图居中
    let translateX = -minX * scale + (width - (maxX - minX) * scale) / 2;
    let translateY = -minY * scale + (height - (maxY - minY) * scale) / 2;

    // 应用缩放和平移
    g.attr("transform", `translate(${translateX},${translateY}) scale(${scale})`);
    // 尚未解決完問題 end
    const imageElements = g.selectAll('image')
        .data(images)
        .enter()
        .append('image')
        .attr('id', d => d.id)
        .attr('xlink:href', d => d.src)
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .on('mouseover', function (event, d) {
            imageOverview.style('display', 'block')
                .html(d.title)
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY + 10}px`);
        })
        .on('mouseout', function () {
            imageOverview.style('display', 'none');
        });
    imageElements.exit().remove();

    // 圖片邊框
    const rects = g.selectAll('rect')
        .data(images)
        .enter()
        .append('rect')
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .style("fill", "none")
        .style("stroke", "black")
        .style("stroke-width", 2);

    rects.exit().remove();

    // 初始缩放状态
    let currentTransform = d3.zoomIdentity;
    // zoom -> 視圖放大縮小
    svg.call(d3.zoom()
        .extent([[0.1, 10], [width, height]])
        .scaleExtent([1, 8])
        .on("zoom", function (event) {
            currentTransform = event.transform;
            g.attr("transform", currentTransform);
        }));
        
    // Brush -> 將圖片匡起來
    const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('start brush', event => {
            if (event.selection) {
                const [[x0, y0], [x1, y1]] = event.selection;
                rects.each(function (d) {
                    const dx = currentTransform.applyX(d.x);
                    const dy = currentTransform.applyY(d.y);
                    const dw = currentTransform.k * d.width;
                    const dh = currentTransform.k * d.height;
                    // 检查刷选框与矩形是否重叠
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
                        )
                    const rect = d3.select(this);
                    rect.style("stroke", overlaps ? "red" : "black");
                });
            }
        })
        .on('end', event => {
            if (!event.selection) {
                // Reset all rects if the brush is cleared
                rects.style("stroke", "black");
            }
        });

    // Toggle brush and zoom on spacebar press
    document.body.onkeyup = function (e) {
        if (e.key === " ") {
            // Clear brush
            d3.selectAll(".brush").on(".brush", null);
            d3.selectAll(".brush").remove();
            const brushGroup = svg.append("g").attr("class", "brush");
            brushGroup.call(brush);
            svg.on(".zoom", null);
        }
        if (e.key === "Escape") {
            // Clear brush
            d3.selectAll(".brush").on(".brush", null);
            d3.selectAll(".brush").remove();
            rects.style("stroke", "black");
            // Re-enable zoom
            svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([1, 8])
                .on("zoom", function (event) {
                    currentTransform = event.transform;
                    g.attr("transform", currentTransform);
                }));
        }
    };
    
}
