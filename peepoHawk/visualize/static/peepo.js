
d3.json("peepo.json", function(error, json) {
    var w = 1200,
        h = 800;

    var circleWidth = 5;

    var fontFamily = 'Bree Serif',
        fontSizeHighlight = '1.5em',
        fontSizeNormal = '1em';

    var palette = {
          "lightgray": "#819090",
          "gray": "#708284",
          "mediumgray": "#536870",
          "darkgray": "#475B62",
          "whitegrey": "#CCC",

          "darkblue": "#0A2933",
          "darkerblue": "#042029",

          "paleryellow": "#FCF4DC",
          "paleyellow": "#EAE3CB",
          "yellow": "#A57706",
          "orange": "#BD3613",
          "red": "#D11C24",
          "pink": "#C61C6F",
          "purple": "#595AB7",
          "blue": "#2176C7",
          "green": "#259286",
          "yellowgreen": "#738A05"
      }

    var vis = d3.select("body")
        .append("svg:svg")
          .attr("viewBox", "0 0 " + w + " " + h )
          .attr("preserveAspectRatio", "xMidYMid meet")
          .call(d3.behavior.zoom().on("zoom", function () {
            vis.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")")
          }))
        .append("g");

    vis.append("defs").selectAll("marker")
        .data(["arrow"])
        .enter().append("marker")
        .attr("id", function(d) { return d; })
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", 0)
        .attr("markerWidth", 9)
        .attr("markerHeight", 9)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .style("fill", palette.whitegrey);

    var force = d3.layout.force()
        .nodes(json.nodes)
        .links([])
        .gravity(0.1)
        .charge(-1000)
        .size([w, h]);

    var edges = [];
    json.links.forEach(function(e) {
        edges.push({
            source: json.nodes[e.source],
            target: json.nodes[e.target]
        });
    });

    json.nodes.forEach(function(n) {
        n.name = n.id;
    });

     var link = vis.selectAll(".link")
            .data(edges)
            .enter().append("line")
              .attr("class", "link")
              .attr("stroke", palette.whitegrey)
              .attr("fill", "none")
              .attr("marker-end", function(d) { return "url(#arrow)"; });

     var node = vis.selectAll("circle.node")
          .data(json.nodes)
          .enter().append("g")
          .attr("class", "node")

          //MOUSEOVER
          .on("mouseover", function(d,i) {
              //CIRCLE
              d3.select(this).selectAll("circle")
              .transition()
              .duration(250)
              .style("cursor", "none")
              .attr("r", circleWidth+3)
              .attr("fill",palette.orange);

              //TEXT
              d3.select(this).select("text")
              .transition()
              .style("cursor", "none")
              .duration(250)
              .style("cursor", "none")
              .attr("font-size","1.5em")
              .attr("x", 15 )
              .attr("y", 5 )
          })

          //MOUSEOUT
          .on("mouseout", function(d,i) {
              //CIRCLE
              d3.select(this).selectAll("circle")
              .transition()
              .duration(250)
              .attr("r", circleWidth)
              .attr("fill",palette.pink);

              //TEXT
              d3.select(this).select("text")
              .transition()
              .duration(250)
              .attr("font-size","1em")
              .attr("x", 8 )
              .attr("y", 4 )
          })
          //DISABLE BACKGROUND PAN WHEN DRAGGING NODE
          .on('mousedown', function(){
            d3.event.stopPropagation();
          })
          .call(force.drag);


        //CIRCLE
        node.append("svg:circle")
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; })
          .attr("r", circleWidth)
          .attr("fill", function(d, i) { return  palette.pink; } )

        //TEXT
        node.append("text")
          .text(function(d, i) { return d.name; })
        .attr("x",    function(d, i) { return circleWidth + 5; })
          .attr("y",            function(d, i) { return circleWidth + 0 })
          .attr("font-family",  "Bree Serif")
          .attr("fill",         function(d, i) {  return  palette.paleryellow;  })
          .attr("font-size",    function(d, i) {  return  "1em"; })
          .attr("text-anchor",  function(d, i) { return  "beginning"; })

        node.append("text")
            .attr("x",    function(d, i) { return circleWidth + 5; })
            .attr("y",            function(d, i) { return circleWidth + 5 })
            .attr("font-family",  "Bree Serif")
            .attr("fill",         function(d, i) {  return  palette.paleryellow;  })
            .attr("font-size",    function(d, i) {  return  "0.5em"; })
            .attr("text-anchor",  function(d, i) { return  "beginning"; })
            .each(function (d) {
                let lines = d.cpd.split("\n");
                for (var i = 0; i < lines.length; i++) {
                     d3.select(this).append("tspan")
                         .attr("dy", 10)
                         .attr("x", circleWidth + 5)
                          .text(lines[i])
                }});

    force.on("tick", function(e) {
      node.attr("transform", function(d, i) {
            return "translate(" + d.x + "," + d.y + ")";
        });

       link.attr("x1", function(d)   { return d.source.x; })
           .attr("y1", function(d)   { return d.source.y; })
           .attr("x2", function(d)   { return d.target.x; })
           .attr("y2", function(d)   { return d.target.y; })
    });

    force.start();
});