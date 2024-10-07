function initScatterPlot(images) {
    // 圖片Prompt
    const imageOverview = d3.select('#tooltip');
    // Main Map
    const svg = d3.select("#scatter")
    const width = svg.attr("width");
    const height = svg.attr("height");
    // Mini Map
    //const mini_svg = d3.select("#scatter-minimap")
    //const mini_width = mini_svg.attr("width");
    //const mini_height = mini_svg.attr("height");

    // 移除現有的 <g> 元素
    svg.selectAll("g").remove();

    const g = svg.append('g');

    // zoom -> 視圖放大縮小
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
                .html(`<img src="${d.src}" width="300" height="300"><br><strong>${d.title}</strong>`)
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY + 10}px`);
        })
        .on('mouseout', function () {
            imageOverview.style('display', 'none');
        });

    imageElements.exit().remove();

    // 圖片邊框
    let rects = g.selectAll('rect')
        .data(images)
        .enter()
        .append('rect')
        .attr('class', 'image-border')
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .style("fill", "none")
        .style("stroke", "black")
        .style("stroke-width", 2);

    rects.exit().remove();
    //  初始放大縮小狀態會改變
    let currentTransform = d3.zoomIdentity;
    let zoom = d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([1, 8])
        .on("zoom", function (event) {
            currentTransform = event.transform
            // 调整图片位置
            console.log("currentTransform.k:" + currentTransform.k)
            g.selectAll('image')
                .attr('x', d => currentTransform.applyX(d.x * currentTransform.k))  // 根据缩放比例调整位置
                .attr('y', d => currentTransform.applyY(d.y * currentTransform.k)) // 根据缩放比例调整位置
                .attr('width', d => d.width * currentTransform.k)
                .attr('height', d => d.height * currentTransform.k);
            g.selectAll('.image-border')
                .attr('x', d => currentTransform.applyX(d.x * currentTransform.k))  // 根据缩放比例调整位置
                .attr('y', d => currentTransform.applyY(d.y * currentTransform.k)) // 根据缩放比例调整位置
                .attr('width', d => d.width * currentTransform.k)
                .attr('height', d => d.height * currentTransform.k);
        })
    svg.call(zoom);

    // Brush -> 將圖片匡起來
    const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('start brush', event => {
            if (event.selection) {
                const [[x0, y0], [x1, y1]] = event.selection;
                rects.each(function (d) {
                    // 初始缩放状态
                    const dx = currentTransform.applyX(d.x * currentTransform.k);
                    const dy = currentTransform.applyY(d.y * currentTransform.k);
                    const dw = currentTransform.k * d.width;
                    const dh = currentTransform.k * d.height

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
            } else {
                const [[x0, y0], [x1, y1]] = event.selection;
                // 获取被框选的图片信息
                const selectedImages = [];
                rects.each(function (d) {
                    const dx = currentTransform.applyX(d.x * currentTransform.k);
                    const dy = currentTransform.applyY(d.y * currentTransform.k);
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

                    if (overlaps) {
                        selectedImages.push({ id: d.id, title: d.title }); // 添加选中图片的id
                    }
                });
                /* */

                // 创建并触发自定义事件，将数据传递给 HTML
                const scatterSelectionEndEvent = new CustomEvent("scatterSelectionEnd", {
                    detail: selectedImages,
                });
                document.dispatchEvent(scatterSelectionEndEvent);


            }

        });

    // Toggle brush and zoo m on spacebar press
    document.body.onkeyup = function (e) {
        if (e.key === " ") {
            e.preventDefault();
            // Clear brush
            d3.selectAll(".brush").remove();
            const brushGroup = svg.append("g").attr("class", "brush");
            brushGroup.call(brush);
            svg.on(".zoom", null);
        }
        if (e.key === "t") {
            // Clear brush
            d3.selectAll(".brush").on(".brush", null);
            d3.selectAll(".brush").remove();
            rects.style("stroke", "black");
            // Re-enable zoom
            svg.call(zoom);

        }
    };

}
