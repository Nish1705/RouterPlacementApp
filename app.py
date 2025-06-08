import streamlit as st
import io
import contextlib
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
from streamlit_ace import st_ace
import numpy as np
from scipy.spatial import ConvexHull
from Heuristic import HeuristicRouterPlacement as H
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def getAbreviation(key):
    l = ['a' if key[0]else '','b'if key[1]else '','c' if key[2] else '', 'd' if key[3] else '']
    x = ''.join(l)
    if len(x)==0:
        ans  = "None"
    elif len(x)==1:
        ans = f"Only {x}"
    else:
        ans = ''
        for i in x:
            ans += i + '+'
        ans = ans.strip('+')
    return ans

    
def generateResult():
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
        # print(f"Heuristic time: {time_heur_temp} seconds")
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
                plt.plot(point[0], point[1], 'gx')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=False, linestyle='dotted')
                plt.gca().add_patch(circle)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot {getAbreviation(tuple(heuristic_options[i]))}",
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
                plt.plot(point[0], point[1], 'gx')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=False, linestyle='dotted')
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot {getAbreviation(tuple(heuristic_options[i+1]))}",
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
                plt.plot(point[0], point[1], 'gx')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=False, linestyle='dotted')
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot {getAbreviation(tuple(heuristic_options[i+2]))}",
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
                plt.plot(point[0], point[1], 'gx')
                circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=False, linestyle='dotted')
                plt.gca().add_patch(circle)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot {getAbreviation(tuple(heuristic_options[i+3]))}",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                use_container_width=True

            )
        
    
            
    
    col1,col2 = st.columns(2)
    with col1:
        st.bar_chart(all_heur)
    with col2:
        st.bar_chart(all_heur_times)

    

        # plt.show()




        
        



def discretize(h_points,NUM_PARTITIONS):    
    # h_points = [[0,0],[0,1],[1,1],[1,0],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0,0.98],[0.02,1],[0.04,1],[0,0.96]]

    # h_points = h_points
    h_points = np.array(h_points)

    # plt.plot(h_points[:,0], h_points[:,1], 'o')
    # plt.show()

    #Convex hull and its bounding box
    hull = ConvexHull(h_points)

    # Bounding box of the convex hull
    min_x = min(h_points[:,0])
    min_y = min(h_points[:,1])
    max_x = max(h_points[:,0])
    max_y = max(h_points[:,1])
    bb = [(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    
    nx, ny = (NUM_PARTITIONS, NUM_PARTITIONS) # number of horizontal and vertical grid lines

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    g = np.meshgrid(x, y) #, sparse=True )
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

# st_autorefresh(interval=50000000000000)



global HEUR_RESULT
HEUR_RESULT =None
st.title("🙌Minimum Router Estimation")

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


generate = st.button("Generate Plot",use_container_width=True)
result = st.container()



if generate:
    # Cache results into session_state
    heur_results, all_heur, all_heur_times, h_points, heuristic_options = generateResult()

    st.session_state.heur_results = heur_results
    st.session_state.all_heur = all_heur
    st.session_state.all_heur_times = all_heur_times
    st.session_state.h_points = h_points
    st.session_state.heuristic_options = heuristic_options
    st.session_state.router_range = ROUTER_RANGE
    st.session_state.scale = SCALE
    st.session_state.N = N
    st.session_state.discretization = NUM_PARTITIONS
    

# Try to use cached results from session_state
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
        
        
        # st.scatter_chart(h_points,use_container_width=True)

  
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

    # st.error(str(e))  # Optional: to help you debug
# dataPresent = False
# if generate:

    
#     if heur_results != None:
#         dataPresent = True
#     else:
#         dataPresent=False
# try:
#     with result:
        
# except:
#     pass
    











        