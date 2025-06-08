import streamlit as st
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from Heuristic import HeuristicRouterPlacement as H
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from streamlit_drawable_canvas import st_canvas


def getAbreviation(key):
    l = ['A' if key[0]else '','B'if key[1]else '','C' if key[2] else '', 'D' if key[3] else '']
    x = ''.join(l)
    if len(x)==0:
        ans  = "Greedy(G)"
    elif len(x)==1:
        ans = f"Only G+{x}"
    else:
        ans = ''
        for i in x:
            ans += i + '+'
        ans = ans.strip('+')
        ans = f"G+{ans}"
    return ans

    
def generateResult(custom_points,node_points):
    global HEUR_RESULT
    heuristic_options = [
        [False,False,False,False],
        [False,False,False,True],
        [False,False,True,False],
        [False,False,True,True],
        [False,True,False,False],
        [False,True,False,True],
        [False,True,True,False],
        [False,True,True,True],
        [True,False,False,False],
        [True,False,False,True],
        [True,False,True,False],
        [True,False,True,True],
        [True,True,False,False],
        [True,True,False,True],
        [True,True,True,False],
        [True,True,True,True]
    ]

    
    if custom_points:
        h_points = np.array(node_points)

    else:
        rng = np.random.default_rng()
        h_points = rng.random((N, 2))
        h_points = np.array(h_points)
        h_points = h_points*SCALE


    coords = discretize(h_points,NUM_PARTITIONS)
    plt.clf()
    all_heur={}
    all_heur_times={}
    heur_results = []

    for i,heuristic in enumerate(heuristic_options):



        start_heur = time.time()
        Heur_model = H(h_points=h_points, coords=coords, router_range=ROUTER_RANGE*SCALE, max_routers=100, heuristics=heuristic)
        HEUR_RESULT = Heur_model.run()
        time_heur_temp =  time.time() - start_heur
        if HEUR_RESULT == None:

            return None,None,None,None,None
        else:
            all_heur[getAbreviation(tuple(heuristic))]=len(HEUR_RESULT)
            all_heur_times[getAbreviation(tuple(heuristic))]=time_heur_temp*1000000
            heur_results.append(tuple(HEUR_RESULT))


    return heur_results,all_heur,all_heur_times,h_points,heuristic_options

def plotResults(heur_results,all_heur,all_heur_times,h_points,heuristic_options,router_range,scale):
    
    for i in range(0,16,4):

        col1,col2,col3,col4 = st.columns(4)
        with col1:
            plt.figure(figsize=(3.3,2.7))
            plt.title(f"{getAbreviation(tuple(heuristic_options[i]))} : {len(heur_results[i])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o')
            for point in heur_results[i]:
                plt.plot(point[0], point[1], 'X',color='black')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                plt.gca().add_patch(circle)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i]))}\"",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                use_container_width=True

            )
            

        with col2:
            plt.figure(figsize=(3.3,2.7))
            plt.title(f"{getAbreviation(tuple(heuristic_options[i+1]))} : {len(heur_results[i+1])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o')
            for point in heur_results[i+1]:
                plt.plot(point[0], point[1], 'X',color='black')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+1]))}\"",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                use_container_width=True
            )
            

        with col3:
            plt.figure(figsize=(3.3,2.7))
            plt.title(f"{getAbreviation(tuple(heuristic_options[i+2]))} : {len(heur_results[i+2])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o')
            for point in heur_results[i+2]:
                plt.plot(point[0], point[1], 'X',color='black')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+2]))}\"",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                use_container_width=True

            )
            
            
        with col4:
            plt.figure(figsize=(3.3,2.7))
            plt.title(f"{getAbreviation(tuple(heuristic_options[i+3]))} : {len(heur_results[i+3])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o')
            for point in heur_results[i+3]:
                plt.plot(point[0], point[1], 'X',color='black')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+3]))}\"",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                use_container_width=True

            )
        
    
    with  st.container(border=1):
        st.subheader("Analytics",help="Time and Result Comparison for each Heuristic Combination")
        try:
            
            col1,col2 = st.columns(2)
            with col1:
                st.bar_chart(all_heur,x_label='Heuristic Combination',y_label='Number of Routers',horizontal=True)
            with col2:
                st.bar_chart(all_heur_times,x_label='Heuristic Combination',y_label='Time (μs)',horizontal=True)
        except:
            st.warning("Something went wrong with the generation of Analytics Graphs")
        
        



def discretize(h_points,NUM_PARTITIONS):    
    h_points = np.array(h_points)
    hull = ConvexHull(h_points)

    min_x = min(h_points[:,0])
    min_y = min(h_points[:,1])
    max_x = max(h_points[:,0])
    max_y = max(h_points[:,1])
    bb = [(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    
    nx, ny = (NUM_PARTITIONS, NUM_PARTITIONS) 

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    g = np.meshgrid(x, y) 
    coords = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)
        

    to_del = []
    def point_in_hull(point, hull, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)

    for i in range(len(coords)):
        p = coords[i]
        point_is_in_hull = point_in_hull(p, hull)
        if (not (point_is_in_hull)):
            to_del.append(i)
        
    coords = np.delete(coords, to_del, 0)

    
    return coords

st.set_page_config(page_title="Router Estimation", layout="wide")



global HEUR_RESULT
HEUR_RESULT =None
st.title("Minimum Router Estimation")


sidebar = st.sidebar
sidebar.title("About")
with sidebar:
    if st.button("ℹ️ About"):
        st.session_state.show_about = not st.session_state.get("show_about", False)

if st.session_state.get("show_about"):
    st.sidebar.markdown("### 📡 Router Placement App")
    st.sidebar.info("""
    This tool helps optimize the placement of routers in a 2D space.

    **Parameters:**
    - Number of nodes
    - Partitions
    - Scale of space
    - Router range

    The app aims to achieve maximum coverage using heuristic placement strategies.
    """)


col1,col2,col3,col4 = st.columns(4)
with col1:
    ROUTER_RANGE = st.selectbox(
        "Choose a router range",
        (0.1, 0.2, 0.3, 0.4, 0.5),accept_new_options=False
    )

with col2:
    N = st.selectbox(
        "Choose Number of Points",
        (5,10,15,20,25,30),accept_new_options=False
    )


with col3:

    NUM_PARTITIONS = st.selectbox(
        "Choose number of Partitions",
        (5,11,15,21,25,31,35,41,45,51),accept_new_options=False
    )
with col4:
    SCALE = st.selectbox(
        "Choose Scale",
        (1,10,100,1000),accept_new_options=False
    )
custom_points=False

show_canvas = st.checkbox("Add Custom Points")


if show_canvas:
    canvas_px = 500
    col1,col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=5,
            background_color="#ffffff",
            height=canvas_px,
            width=canvas_px,
            drawing_mode="point",
            key="canvas_bl",
        )

    with col2:

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                node_points = [
                    (
                        round(obj["left"] / canvas_px * SCALE, 2),  # X stays the same
                        round((canvas_px - obj["top"]) / canvas_px * SCALE, 2)  # Flip Y
                    )
                    for obj in objects if obj["type"] == "circle"
                ]
                
                custom_points = True
                df_nodes = pd.DataFrame(node_points, columns=["x", "y"])
                st.write(f"### 📍 Selected Locations (Scale: {SCALE}×{SCALE})")
                st.dataframe(df_nodes)
            else:
                st.info("Click on the canvas to place nodes.")
        else:
            st.info("Draw on the canvas to add points.")


generate = st.button("Generate Plot",use_container_width=True)
result = st.container()

if custom_points:
    node_points =  np.array(node_points)
    print(node_points)
    N = len(node_points)
else:
    node_points = None




if generate:
    heur_results, all_heur, all_heur_times, h_points, heuristic_options = generateResult(custom_points,node_points)

    st.session_state.heur_results = heur_results
    st.session_state.all_heur = all_heur
    st.session_state.all_heur_times = all_heur_times
    st.session_state.h_points = h_points
    st.session_state.heuristic_options = heuristic_options
    st.session_state.router_range = ROUTER_RANGE
    st.session_state.scale = SCALE
    st.session_state.N = N
    st.session_state.discretization = NUM_PARTITIONS
    

try:

    if all(
        key in st.session_state for key in [
            "heur_results", "all_heur", "all_heur_times", "h_points", "heuristic_options",'router_range','scale','N','discretization'
        ]
    ):
        with st.container(border=5):
            col2,col3,col4,col5 = st.columns(4)
            with col2:
                st.info(f"Number of Points : {st.session_state.N}")
            with col3:
                st.info(f"Discretization : {st.session_state.discretization}")
            with col4:
                st.info(f"Scale : {st.session_state.scale} x {st.session_state.scale}")
            with col5:
                st.info(f"Router Range : {st.session_state.router_range*st.session_state.scale} m")
  
            plotResults(
                st.session_state.heur_results,
                st.session_state.all_heur,
                st.session_state.all_heur_times,
                st.session_state.h_points,
                st.session_state.heuristic_options,
                st.session_state.router_range,
                st.session_state.scale
            )


    
except Exception as e:
    col1,col2 = st.columns(2)
    with col1:
        st.info("No Valid Result")
    with col2:
        st.info("Try increasing Number of Paritions for a valid result")


    











        