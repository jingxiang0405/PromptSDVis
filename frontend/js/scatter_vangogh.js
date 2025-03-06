function initScatterPlot(images) {
    console.log('123231321')
    // Main Map
    const svg = d3.select("#scatter");
    const width = +svg.attr("width");
    const height = +svg.attr("height");

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

    // 移除主地圖上舊的 g
    svg.selectAll("g.main-group").remove();
    const g = svg.append('g').attr("class", "main-group");

    // 將資料座標映射到主地圖範圍內
    // 初始位置已縮放到主地圖大小內
    const imageElements = g.selectAll('image')
        .data(images)
        .enter()
        .append('image')
        .attr('xlink:href', d => d.src)
        .attr('x', d => xScaleMain(d.x))
        .attr('y', d => yScaleMain(d.y))
        .attr('width', d => d.width)
        .attr('height', d => d.height);

    imageElements.exit().remove();

    // 繪製邊框 rect（圖片邊界）
    let rects = g.selectAll('rect')
        .data(images)
        .enter()
        .append('rect')
        .attr('data-original-stroke', d => d.color) // 用 label 的顏色作為邊框顏色
        .attr('class', 'image-border')
        .attr('x', d => xScaleMain(d.x))
        .attr('y', d => yScaleMain(d.y))
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .style("fill", "none")
        .style("stroke", d => d.color) // 邊框使用對應的顏色
        .style("stroke-width", 4);

    rects.exit().remove();

    let currentTransform = d3.zoomIdentity;

    // 定義 zoom 行為
    let zoom = d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([-2, 15])
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
        });

    svg.call(zoom);
}
