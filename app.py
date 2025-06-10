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
import zipfile
from PIL import Image
import math

FIG_SIZE = (3.3,2.7)

def initPlotParams(title=None, xlabel=None, ylabel=None, color='skyblue', edgecolor='black', alpha=1.0, grid=False,equal=False):
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "sans-serif" # Set default to serif
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"] # Set fallback fonts for sans-serif
    # plt.figure(figsize=(3.3,2.7)) # Width and height in inches
    if equal:
        plt.axis('equal')


    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.grid(grid)


def getAbreviation(key):
    l = ['A' if key[0]else '','B'if key[1]else '','C' if key[2] else '', 'D' if key[3] else '']
    x = ''.join(l)
    if len(x)==0:
        ans  = "Greedy(G)"
    elif len(x)==1:
        ans = f"G+{x}"
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


    coords,hull = discretize(h_points,NUM_PARTITIONS)

    initial_plots = []


    plt.figure(figsize=FIG_SIZE)
    initPlotParams(equal=True)
    
    plt.plot(h_points[:,0], h_points[:,1], 'bo')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    initial_plots.append(buf.getvalue())
    plt.close()

    if hull is not None:
        plt.figure(figsize=FIG_SIZE)
        initPlotParams(equal=True)

        plt.plot(h_points[:,0], h_points[:,1], 'bo',label='Regular Nodes')
        for simplex in hull.simplices:
            plt.plot(h_points[simplex, 0], h_points[simplex, 1], 'k-')
        for p in coords:
            marker = '.' 
            color = 'g'
            plt.scatter(p[0], p[1], marker=marker, color=color,label="Grid Points" if p[0] == coords[0][0] and p[1] == coords[0][1] else "")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        initial_plots.append(buf.getvalue())

        plt.close()
    else:
        plt.figure(figsize=FIG_SIZE)
        initPlotParams(equal=True)

        plt.plot(h_points[:,0], h_points[:,1], 'bo',label='Regular Nodes')

        for p in coords:
            marker = '.' 
            color = 'g'
            plt.scatter(p[0], p[1], marker=marker, color=color,label="Grid Points" if p[0] == coords[0][0] and p[1] == coords[0][1] else "")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        initial_plots.append(buf.getvalue())
        plt.close()



    all_heur={}
    all_heur_times={}
    heur_results = []
    intermediates = {}
    res_without_clean = {
        

    }
    

    for i,heuristic in enumerate(heuristic_options):

        if heuristic[3]==True:
            withoutClean = [heuristic[0],heuristic[1],heuristic[2],False]
            routers,time_heur_temp = res_without_clean[getAbreviation(tuple(withoutClean))]

            start_heur = time.time()
            Heur_model = H(h_points=h_points, coords=coords, router_range=ROUTER_RANGE*SCALE, max_routers=100, heuristics=heuristic)
            HEUR_RESULT = Heur_model.clean(nodes=h_points,routers=routers,r=ROUTER_RANGE*SCALE)
            
            time_heur_temp +=  time.time() - start_heur

            if HEUR_RESULT ==None:
                return None,None,None,None,None,None
            else:
                all_heur[getAbreviation(tuple(heuristic))]=len(HEUR_RESULT)
                all_heur_times[getAbreviation(tuple(heuristic))]=time_heur_temp*1000
                intermediates[getAbreviation(tuple(heuristic))] = intermediates[getAbreviation(tuple(withoutClean))]
                heur_results.append(tuple(HEUR_RESULT))


        else:
            start_heur = time.time()
            Heur_model = H(h_points=h_points, coords=coords, router_range=ROUTER_RANGE*SCALE, max_routers=100, heuristics=heuristic)
            HEUR_RESULT,inter = Heur_model.run()            
            time_heur_temp =  time.time() - start_heur

            if HEUR_RESULT ==None:
                return None,None,None,None,None,None
            else:
                all_heur[getAbreviation(tuple(heuristic))]=len(HEUR_RESULT)
                all_heur_times[getAbreviation(tuple(heuristic))]=time_heur_temp*1000
                intermediates[getAbreviation(tuple(heuristic))] = [initial_plots[0],initial_plots[1],*inter]
                heur_results.append(tuple(HEUR_RESULT))
                res_without_clean[getAbreviation(tuple(heuristic))] =[HEUR_RESULT,time_heur_temp]


    print(all_heur_times)
    return heur_results,all_heur,all_heur_times,h_points,heuristic_options,intermediates
                  
def plotIntermediates(inter,download_format,heuristic_options):
    save_format = 'png' if download_format == 'PNG' else 'pdf'
    # print(len(inter))

    with st.expander("Show Intermediates"):
        st.markdown('## Intermediates')


        tab_keys = list(inter.keys())
        tabs = st.tabs(tab_keys)

        for tab_idx, tab in enumerate(tabs):
            key = tab_keys[tab_idx]
            buffers = inter[key]

            with tab:
                # st.info(f"Intermediate Steps for: **{key}**")

                # Group buffers into chunks of 4 for rows
                for i in range(0, len(buffers), 4):
                    cols = st.columns(4)
                    for j in range(4):
                        if i + j < len(buffers):
                            with cols[j]:
                                img = Image.open(io.BytesIO(buffers[i + j]))
                                st.image(img, use_container_width=True)
                                st.download_button(
                                label=f"Download Plot: \"{key}_{i+j}\"",
                                data=buffers[i + j],
                                file_name=f"plot_{key}_{i+j}.{save_format}",
                                mime=f"image/{save_format}",
                                use_container_width=True

                            )


def plotResults(heur_results,all_heur,all_heur_times,h_points,heuristic_options,router_range,scale,download_format):

    all_buffers = []
    for i in range(0,16,4):

        col1,col2,col3,col4 = st.columns(4)
        with col1:
            plt.figure(figsize=FIG_SIZE)
            initPlotParams(equal=True)

            # plt.title(f"{getAbreviation(tuple(heuristic_options[i]))} : {len(heur_results[i])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o',zorder=-1)
            for point in heur_results[i]:
                plt.plot(point[0], point[1], 'X',color='black')
                # circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                # plt.gca().add_patch(circle)

            xs, ys = zip(*heur_results[i])
            # plt.scatter(xs, ys, color='blue', label='Routers')

            # Draw undirected edges between routers within range r
            for k, (x1, y1) in enumerate(heur_results[i]):
                for j in range(k + 1, len(heur_results[i])):
                    x2, y2 = heur_results[i][j]
                    if math.dist((x1, y1), (x2, y2)) <= router_range*scale:
                            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5)

            plt.tight_layout()

            buf = io.BytesIO()
            save_format = 'png' if download_format == 'PNG' else 'pdf'
            plt.savefig(buf, format=save_format)
            buf.seek(0)
            filename_ext = save_format

            all_buffers.append((f"plot_{getAbreviation(tuple(heuristic_options[i]))}.{filename_ext}", buf.getvalue()))

            
            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i]))}\"",
                data=buf,
                file_name=f"plot_{getAbreviation(tuple(heuristic_options[i]))}.{save_format}",
                mime=f"image/{filename_ext}",
                use_container_width=True

            )
        
            

        with col2:
            plt.figure(figsize=FIG_SIZE)
            initPlotParams(equal=True)


            # plt.title(f"{getAbreviation(tuple(heuristic_options[i+1]))} : {len(heur_results[i+1])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o',zorder=-1)
            for point in heur_results[i+1]:
                plt.plot(point[0], point[1], 'X',color='black')
                # circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                # plt.gca().add_patch(circle)
            
            xs, ys = zip(*heur_results[i+1])
            # plt.scatter(xs, ys, color='blue', label='Routers')

            # Draw undirected edges between routers within range r
            for k, (x1, y1) in enumerate(heur_results[i+1]):
                for j in range(k + 1, len(heur_results[i+1])):
                    x2, y2 = heur_results[i+1][j]
                    if math.dist((x1, y1), (x2, y2)) <= router_range*scale:
                            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5)

                

            plt.tight_layout()

            buf = io.BytesIO()
            save_format = 'png' if download_format == 'PNG' else 'pdf'
            plt.savefig(buf, format=save_format)
            buf.seek(0)
            filename_ext = save_format

            all_buffers.append((f"plot_{getAbreviation(tuple(heuristic_options[i+1]))}.{filename_ext}", buf.getvalue()))

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+1]))}\"",
                data=buf,
                file_name=f"plot_{getAbreviation(tuple(heuristic_options[i+1]))}.{filename_ext}",
                mime=f"image/{filename_ext}",
                use_container_width=True
            )

        with col3:
            plt.figure(figsize=FIG_SIZE)
            initPlotParams(equal=True)
            

            # plt.title(f"{getAbreviation(tuple(heuristic_options[i+2]))} : {len(heur_results[i+2])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o',zorder=-1)
            for point in heur_results[i+2]:
                plt.plot(point[0], point[1], 'X',color='black')
                # circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                # plt.gca().add_patch(circle)
            xs, ys = zip(*heur_results[i+2])
            # plt.scatter(xs, ys, color='blue', label='Routers')

            # Draw undirected edges between routers within range r
            for k, (x1, y1) in enumerate(heur_results[i+2]):
                for j in range(k + 1, len(heur_results[i+2])):
                    x2, y2 = heur_results[i+2][j]
                    if math.dist((x1, y1), (x2, y2)) <= router_range*scale:
                            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5)


            plt.tight_layout()

            buf = io.BytesIO()
            save_format = 'png' if download_format == 'PNG' else 'pdf'
            plt.savefig(buf, format=save_format)
            buf.seek(0)
            filename_ext = save_format


            all_buffers.append((f"plot_{getAbreviation(tuple(heuristic_options[i+2]))}.{filename_ext}", buf.getvalue()))

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+2]))}\"",
                data=buf,
                file_name=f"plot_{getAbreviation(tuple(heuristic_options[i+2]))}.{filename_ext}",
                mime=f"image/{filename_ext}",
                use_container_width=True

            )
            
            
        with col4:
            
            plt.figure(figsize=FIG_SIZE)
            initPlotParams(equal=True)
            # plt.title(f"{getAbreviation(tuple(heuristic_options[i+3]))} : {len(heur_results[i+3])}")

            plt.plot(h_points[:,0], h_points[:,1], 'o',zorder=-1)
            for point in heur_results[i+3]:
                plt.plot(point[0], point[1], 'X',color='black')
                # circle = plt.Circle((point[0], point[1]), router_range*scale , color='green', fill=True, linestyle='dotted',alpha=0.1)
                # plt.gca().add_patch(circle)
            
            xs, ys = zip(*heur_results[i+3])
            # plt.scatter(xs, ys, color='blue', label='Routers')

            # Draw undirected edges between routers within range r
            for k, (x1, y1) in enumerate(heur_results[i+3]):
                for j in range(k + 1, len(heur_results[i+3])):
                    x2, y2 = heur_results[i+3][j]
                    if math.dist((x1, y1), (x2, y2)) <= router_range*scale:
                            plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5)

            plt.tight_layout()

            buf = io.BytesIO()
            save_format = 'png' if download_format == 'PNG' else 'pdf'
            plt.savefig(buf, format=save_format)
            buf.seek(0)
            filename_ext = save_format

            all_buffers.append((f"plot_{getAbreviation(tuple(heuristic_options[i+3]))}.{filename_ext}", buf.getvalue()))

            st.pyplot(plt)
            st.download_button(
                label=f"Download Plot: \"{getAbreviation(tuple(heuristic_options[i+3]))}\"",
                data=buf,
                file_name=f"plot_{getAbreviation(tuple(heuristic_options[i+3]))}.{filename_ext}",
                mime=f"image/{filename_ext}",
                use_container_width=True

            )
        st.markdown("---")  
        

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zip_file:
        for filename, data in all_buffers:
            zip_file.writestr(filename, data)
    zip_buf.seek(0)


    # Download button
    st.download_button(
        label=f"📦 Download All Plots as {download_format} (ZIP)",
        data=zip_buf,
        file_name=f"all_plots.{download_format.lower()}.zip",
        mime="application/zip",
        use_container_width=True,
        help=f'Download all the plots in {download_format} format as a ZIP file.'
    )


        
    
    with  st.container(border=1):
        st.subheader("Analytics",help="Time and Result Comparison for each Heuristic Combination")
        try:

            col1,col2 = st.columns(2)

            with col1:
                try:
                    plt.figure(figsize=(10,6))             
                    initPlotParams(ylabel = 'Heuristic',xlabel='Minimized Number of Routers')
                    bars = plt.barh(list(all_heur.keys()),all_heur.values(),color='crimson')
                    for bar in bars:
                        plt.text(bar.get_width()+0.1,
                                  bar.get_y()+bar.get_height()/2,
                                  "{}".format(bar.get_width()),
                                  va='center',
                                  ha='left'
                                  )
                    plt.tight_layout()
                    st.pyplot(plt)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close()



                    st.download_button(
                                label=f"Download Result Analytics Graphs",
                                data=buf,
                                file_name=f"Result_Analytics.{save_format}",
                                mime=f"image/{save_format}",
                                use_container_width=True

                            )

                    # st.bar_chart(all_heur,x_label='Heuristic Combination',y_label='Number of Routers',horizontal=True)
                except:
                    st.warning(body="Something went wrong with the generation of Result Analytics Graphs",icon="⚠️")

            with col2:
                try:
                
                    plt.figure(figsize=(10,6))  
                    initPlotParams(ylabel = 'Heuristic',xlabel='Time (in ms)')
                    bars = plt.barh(list(all_heur_times.keys()),all_heur_times.values(),color='navy')
                    for bar in bars:
                        plt.text(bar.get_width()//2,
                                  bar.get_y()+bar.get_height()/2,
                                  "{}".format(bar.get_width().round(2)),
                                  va='center',
                                  ha='center',
                                  color='white'
                                  )
                    plt.tight_layout()
                    st.pyplot(plt)

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close()



                    st.download_button(
                                label=f"Download Time Analytics Graphs",
                                data=buf,
                                file_name=f"Time_Analytics.{save_format}",
                                mime=f"image/{save_format}",
                                use_container_width=True

                            )
                except:
                    st.warning(body="Something went wrong with the generation of Time Analytics Graphs",icon="⚠️")

        except Exception as e:
            st.warning("Something went wrong with the generation of Analytics Graphs")
            st.error(e)
            print(e)
        
        



def discretize(h_points,NUM_PARTITIONS): 
    if len(h_points)<2:
        return None,None
    if len(h_points) == 2:
        x1, y1 = h_points[0]
        x2, y2 = h_points[1]

        t_values = np.linspace(0, 1, NUM_PARTITIONS + 1).reshape(-1, 1)
        coords = (1 - t_values) * h_points[0] + t_values * h_points[1]

        return coords,None

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

    


    
    return coords,hull

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


col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    range_perc = st.selectbox(
        "Choose a router range",
        (10, 20, 30, 40, 50),accept_new_options=False,
        help="Calculated as x\% of SCALE (in meters)"
    )
    ROUTER_RANGE = range_perc/100

with col2:
    # N = st.selectbox(
    #     "Choose Number of Points",
    #     (5,10,15,20,25,30),accept_new_options=False
    # )
    N = st.number_input(label="Number of Nodes",
                        min_value=2,
                        help="Define the Number of Nodes you need to place on the field.\
                              These nodes are randomly generated."
                        )


with col3:

    NUM_PARTITIONS = st.selectbox(
        "Choose number of Partitions",
        (5,11,15,21,25,31,35,41,45,51,55,61,65,71,75,81,85,91,95,101),accept_new_options=False
    )
with col4:
    SCALE = st.selectbox(
        "Choose Scale",
        (1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000),
        help="This will define the area of the field.\n {Field will be scale x scale}"
    )
        
with col5:
    download_format = st.radio("Plot Download Format:", ["PNG", "PDF"], 
                               horizontal=True,
                               help="Mention the Format of the plots"
                               )

custom_points=False

method_container = st.container(border=2)
with method_container:
    show_canvas = st.checkbox("Add Custom Points")
    if show_canvas:
        
        method = st.selectbox(
            "Choose a method to add custom points",
            ("Create on Canvas", "Upload CSV"),accept_new_options=False  #TODO Adding "Manual" option
        )
        if method == "Create on Canvas":
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
                        temp_points = [
                            (
                                max(round(obj["left"] / canvas_px * SCALE, 2),0),  # X stays the same
                                max(0,round((canvas_px - obj["top"]) / canvas_px *SCALE, 2))  # Flip Y
                            )
                            for obj in objects if obj["type"] == "circle"
                        ]
                        node_points = [
                            (
                                round(obj["left"] / canvas_px , 2),  # X stays the same
                                round((canvas_px - obj["top"]) / canvas_px , 2)  # Flip Y
                            )
                            for obj in objects if obj["type"] == "circle"
                        ]
                        
                        custom_points = True
                        df_nodes = pd.DataFrame(temp_points, columns=["x", "y"])
                        st.write(f"### 📍 Selected Locations (Scale: {SCALE}×{SCALE})")
                        st.dataframe(df_nodes)
                    else:
                        st.info("Click on the canvas to place nodes.")
                else:
                    st.info("Draw on the canvas to add points.")
        elif method == "Upload CSV":
            try:
                uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    df_nodes = df[['x', 'y']]
                    node_points = df_nodes.values.tolist()
                    custom_points = True
                    st.dataframe(df)
                    st.success("Node Points uploaded successfully!")
            except:
                st.error("CSV MUST Contain columns x and y.")
        else:
            pass


generate = st.button("Generate Plot",use_container_width=True)








if generate: 
        if custom_points:
            if len(node_points)>1:
                    
                node_points =  np.array(node_points)*SCALE
                N = len(node_points)
            else:
                st.warning("Atleast 2 Nodes needed for the simulation...\n" \
                        "Continuing with Random Points"
                        )
                custom_points = False
                node_points = None
        else:
            node_points = None            
        
        with st.spinner("🌀 Calculating optimal router positions..."):
            heur_results, all_heur, all_heur_times, h_points, heuristic_options, inter = generateResult(custom_points,node_points)
            if heur_results is not None:
                st.session_state.heur_results = heur_results
                st.session_state.all_heur = all_heur
                st.session_state.all_heur_times = all_heur_times
                st.session_state.h_points = h_points
                st.session_state.heuristic_options = heuristic_options
                st.session_state.router_range = ROUTER_RANGE
                st.session_state.scale = SCALE
                st.session_state.N = N
                st.session_state.discretization = NUM_PARTITIONS
                st.session_state.intermediates = inter
        
                st.success("✔️ Computation Complete!")
            else:
                try:
                    del st.session_state.heur_results
                    del st.session_state.all_heur
                    del st.session_state.all_heur_times
                    del st.session_state.h_points
                    del st.session_state.heuristic_options
                    del st.session_state.router_range
                    del st.session_state.scale
                    del st.session_state.N
                    del st.session_state.discretization
                    del st.session_state.intermediates
                except:
                    pass
                st.warning("⚠️ No Valid Result")



try:
    if not custom_points:
        with st.container(border=1,height=300):
            st.markdown("#### Random Points For Simulation: ")
            df = pd.DataFrame(st.session_state.h_points, columns=["x", "y"])
            st.table(data=df[["x", "y"]])
    


    st.session_state.download_format = download_format

    if all(
        key in st.session_state for key in [
            "heur_results", "all_heur", "all_heur_times", "h_points", "heuristic_options",'router_range','scale','N','discretization','download_format','intermediates'
        ]
    ):
        with st.container(border=1):
            col1,col2,col3,col4,col5 = st.columns(5)
            with col1:
                st.info(f"Download Format : {st.session_state.download_format}")
            with col2:
                st.info(f"Number of Points : {st.session_state.N}")
            with col3:
                st.info(f"Discretization : {st.session_state.discretization}")
            with col4:
                st.info(f"Area : {st.session_state.scale} m × {st.session_state.scale} m")
            with col5:
                st.info(f"Router Range : {st.session_state.router_range*st.session_state.scale} m")
            if st.session_state.intermediates is not None:
                try:
                    plotIntermediates(
                        # st.session_state.heur_results,
                        # st.session_state.all_heur,
                        # st.session_state.all_heur_times,
                        # st.session_state.h_points,
                        # st.session_state.router_range,
                        # st.session_state.scale,
                        st.session_state.intermediates,
                        st.session_state.download_format,
                        st.session_state.heuristic_options
                    )
                except Exception as e:
                    st.warning("⚠️ Error Occured While Generating Intermediates")
                    st.error(e)
            st.markdown("---")
            st.markdown("## Results")
            plotResults(
                st.session_state.heur_results,
                st.session_state.all_heur,
                st.session_state.all_heur_times,
                st.session_state.h_points,
                st.session_state.heuristic_options,
                st.session_state.router_range,
                st.session_state.scale,
                st.session_state.download_format
            )

            


    
except Exception as e:
    col1,col2 = st.columns(2)
    with col1:
        st.info("No Valid Result")
    with col2:
        st.info("Try increasing Number of Paritions for a valid result")
    st.error(e)



    











        