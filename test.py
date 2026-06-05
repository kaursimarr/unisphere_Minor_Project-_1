import streamlit as st
import streamlit.components.v1 as components
import json

st.set_page_config(layout="wide")

# Test coordinates for Seoul National University
uni = {
    "university": "Seoul National University",
    "latitude": 37.5665,
    "longitude": 126.978,
    "rank": 1
}

globe_points = [uni]
globe_json = json.dumps(globe_points)
selected_json = json.dumps(uni["university"])

html = f"""
<div id="globe" style="width:100%;height:500px;"></div>

<script>
const script1 = document.createElement("script");
script1.src = "https://fastly.jsdelivr.net/npm/echarts@5/dist/echarts.min.js";
script1.onload = () => {{
    const script2 = document.createElement("script");
    script2.src = "https://fastly.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js";
    script2.onload = () => {{

        window.globeData = {globe_json};
        window.selectedUni = {selected_json};

        var chart = echarts.init(document.getElementById('globe'));

        const sel = window.globeData[0];

        chart.setOption({{
            backgroundColor: '#f7f9fc',
            globe: {{
                baseTexture: 'https://fastly.jsdelivr.net/gh/ecomfe/echarts-examples/public/data-gl/asset/world.topo.bathy.200401.jpg',
                heightTexture:'https://fastly.jsdelivr.net/gh/ecomfe/echarts-examples/public/data-gl/asset/world.topo.bathy.200401.jpg',
                displacementScale: 0.04,
                shading: 'realistic',
                viewControl: {{
                    autoRotate: true,
                    autoRotateSpeed: 4,
                    distance: 140
                }}
            }},
            series: [
                {{
                    type: 'lines3D',
                    coordinateSystem: 'globe',
                    blendMode: 'lighter',
                    effect: {{
                        show: true,
                        trailWidth: 3,
                        trailLength: 0.25,
                        trailOpacity: 0.95,
                        trailColor: '#22d3ee',
                        period: 4
                    }},
                    lineStyle: {{
                        width: 2,
                        color: '#22d3ee',
                        opacity: 1
                    }},
                    data: [{{
                        coords: [
                            [78.9629, 20.5937],            // India lon, lat
                            [sel.longitude, sel.latitude]  // Seoul lon, lat
                        ]
                    }}]
                }},
                {{
                    type: 'scatter3D',
                    coordinateSystem: 'globe',
                    symbolSize: 12,
                    itemStyle: {{ color: '#f43f5e' }},
                    label: {{
                        show: true,
                        formatter: p => p.data.university,
                        color: '#fff',
                        backgroundColor: 'rgba(0,0,0,0.6)',
                        padding: [4, 6],
                        borderRadius: 4
                    }},
                    data: [
                        {{
                            value: [sel.longitude, sel.latitude, 5],
                            university: sel.university
                        }}
                    ]
                }}
            ]
        }});
    }};
    document.body.appendChild(script2);
}};
document.body.appendChild(script1);
</script>
"""

components.html(html, height=520)
