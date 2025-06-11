'''

Added logic to connect Routers more efficiently
DONE

Limitations: Corner cases are not always satisfied due to limitation in discretization (computationally intensive)
'''

from scipy.spatial import KDTree
import numpy as np
import math 
import networkx as netx
import random
import matplotlib.pyplot as plt
import io
from scipy.spatial import ConvexHull

'''Heuristics :
heuristics[0] : If True, only consider locations that have at least one prior1 node in range
heuristics[1] : If True, maximize in-range distance ONLY
heuristics[2] : If True, minimize out-of-range distance ONLY
If both heuristics[1] and heuristics[2] are True, it will maximize in-range distance and minimize out-of-range distance
heuristics[3] : If True, clean the routers to remove redundant ones
'''

class HeuristicRouterPlacement:

    def __init__(self, h_points, coords, router_range, max_routers, heuristics,scale):
        self.h_points = h_points
        self.coords = coords
        self.router_range = router_range*scale
        self.max_routers = max_routers
        self.heuristics = heuristics
        self.scale = scale
        

    def initPlotParams(self,title=None, xlabel=None, ylabel=None, color='skyblue', edgecolor='black', alpha=1.0, grid=False):
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        plt.rcParams["font.family"] = "sans-serif" # Set default to serif
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"] # Set fallback fonts for sans-serif
        # plt.figure(figsize=(3,3)) # Width and height in inches
        plt.axis('equal')
        plt.xlim(0,1*self.scale)
        plt.ylim(0,1*self.scale)


        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.grid(grid)

        
    def initialCheck(self,h_points,coords,router_range):
        for point in h_points:
            if not any(math.dist(point, possible) <= router_range for possible in coords):
                print("Not Possible to connect all the Source points ")
                return False
        return True
    
    def clean(self,nodes, routers, r):
        if len(routers) <= 1:
                return routers
        tree = KDTree(routers)
        G = netx.Graph()
        
        
        G.add_nodes_from(range(len(routers)))
        

        for i, router in enumerate(routers):
            idxs = tree.query_ball_point(router, r)
            for j in idxs:
                if i < j:  # avoid duplicate edges
                    G.add_edge(i, j)


        node_tree = KDTree(routers)
        node_to_routers = [node_tree.query_ball_point(node, r) for node in nodes]

        def all_nodes_covered(active_router_indices):
                for router_idxs in node_to_routers:
                    if not any(idx in active_router_indices for idx in router_idxs):
                        return False
                return True

        # Start with all routers active
        active_indices = set(range(len(routers)))

        # while True:
        # Compute degrees in the subgraph
        subgraph = G.subgraph(active_indices)
        degrees = subgraph.degree()
        
        # Sort routers by degree (descending)
        sorted_by_degree = sorted(degrees, key=lambda x: x[1])

        for idx, _ in sorted_by_degree:
            temp_indices = active_indices - {idx}
            temp_G = G.subgraph(temp_indices)
            if temp_G:
                if netx.is_connected(temp_G) and all_nodes_covered(temp_indices):
                    active_indices = temp_indices
                    # removed = True
                    # break  # Restart with updated set

        

        # Return the remaining active routers
        return [routers[i] for i in sorted(active_indices)]



        

    def getConnectedComponents(routers, range_value):
        def are_connected(p1, p2):
            return math.dist(p1, p2) <= range_value

        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in routers:
                if neighbor not in visited and are_connected(node, neighbor):
                    dfs(neighbor, component)

        for router in routers:
            if router not in visited:
                component = []
                dfs(router, component)
                components.append(component)

        return len(components), components


    def getOptimalPoint(self,possible_loc, prior1, prior2, range_value):
        
        all_nodes = [*prior1]
        valid_location_dict = {}
        valid_locs = []
        
        for loc in possible_loc:
            if tuple(loc) not in all_routers:
                valid_locs.append(loc)
        possible_loc = valid_locs

        valid_locs = []


        for node in prior2:
            if tuple(node) not in prior1:
                if not any(math.dist(node, p1) <= range_value for p1 in prior1):
                    all_nodes.append(node)
        # Step 1: Filter locations that have at least one prior1 node in range

        if self.heuristics[0]==True:
            for loc in possible_loc:
                if any(math.dist(loc, p1) <= range_value for p1 in prior1):
                    if loc not in all_routers:
                        valid_location_dict[loc] = sum(1 for p1 in prior1 if math.dist(loc, p1) <= range_value)
                        valid_locs.append(loc)

            try:
                max_routers_in_range = max(valid_location_dict.values())
                valid_locs = []
                for loc in valid_location_dict.keys():
                    if valid_location_dict[loc] > 0:
                        valid_locs.append(loc)
            except:
                pass

            # If no locations passed the filter, fallback to all possible locations
            if not valid_locs:
                valid_locs = possible_loc
        else:
            valid_locs = possible_loc

        
        if valid_locs == []:
            return None

        
        



        # Step 2: Calculate in-range and out-of-range distances for each valid location

        in_range_dist = {}
        out_of_range_dist = {}

        tree = KDTree(all_nodes)
        for loc in valid_locs:
            # Get nodes within range
            indices = tree.query_ball_point(loc, range_value)
            
            in_nodes = np.array(all_nodes)[indices]
            in_sum = sum(math.dist(loc, node) for node in in_nodes)
            
            # Get out-of-range nodes (by skipping in-range indices)
            out_sum = sum(math.dist(loc, node) for i, node in enumerate(all_nodes) if i not in indices)
            
            in_range_dist[tuple(loc)] = in_sum
            out_of_range_dist[tuple(loc)] = out_sum



        # Step 3: Find the point that 
        if self.heuristics[1] == True and self.heuristics[2]==True:
            ''' max(in-range distance) && min(out-of-range distance) '''

            best_point = None
            best_score = float('-inf')

            for loc in valid_locs:
                key = tuple(loc)
                score = in_range_dist[key] - out_of_range_dist[key]
                if score > best_score:
                    best_score = score
                    best_point = loc


        elif self.heuristics[1]:

            best_point = None
            best_score = float('-inf')

            for loc in valid_locs:
                key = tuple(loc)
                score = in_range_dist[key]
                if score > best_score:
                    best_score = score
                    best_point = loc




        elif self.heuristics[2]:



            best_point = None
            best_score = float('inf')

            for loc in valid_locs:
                key = tuple(loc)
                score = out_of_range_dist[key]
                if score < best_score:
                    best_score = score
                    best_point = loc

        else:
    
            best_point = random.choice(valid_locs)

        
        

        return best_point
    


    def addMoreRouters(self,h_points,coords,range,max_routers,all_routers):
        global all_buffers
        not_connected = all_routers.copy()  #np.array([])
        connected= []
        k=max_routers
        router_range = range
        router_location = []
        not_connected = np.array(not_connected)

        num_components, components = HeuristicRouterPlacement.getConnectedComponents(all_routers, router_range)

        central_points = []
           

        for i, component in enumerate(components):
            other_points = [pt for j, comp in enumerate(components) if j != i for pt in comp]

            best_point = None
            min_total_dist = float('inf')
            
            for pt in component:
                total_dist = sum(math.dist(pt, other) for other in other_points)
                
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_point = pt

            
            for pt in component:
                if pt != best_point:

                    connected.append(tuple(pt))
                    not_connected = np.delete(not_connected, np.where(np.all(not_connected == pt, axis=1)), axis=0)
                    
                else:
                    pass
                    

            central_points.append(best_point)


        h_points = not_connected.copy()

        central_points = np.array(central_points)

        try:
            hull = ConvexHull(central_points)

            # Bounding box of the convex hull
            min_x = min(central_points[:,0])
            min_y = min(central_points[:,1])
            max_x = max(central_points[:,0])
            max_y = max(central_points[:,1])


            bb = [(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]

            # plt.subplot(rows,cols,subplots)
            plt.figure(figsize=(3.3,2.7))
            
            self.initPlotParams()


            for b in bb:
                plt.plot(b[0], b[1], 'ro',label="Bounding Box" if b==bb[0] else "")

            plt.plot(h_points[:,0], h_points[:,1], 'o',label='Regular Nodes',zorder=1)
            for simplex in hull.simplices:
                plt.plot(central_points[simplex, 0], central_points[simplex, 1], 'k-')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            all_buffers.append(buf.getvalue())
            plt.close()


        except Exception as e:
            cur_points = central_points.copy()
            added = np.array([])

            for point in central_points:
                cur_points = np.vstack((cur_points,point+0.01))
                cur_points = np.vstack((cur_points,point-0.01))
                added = np.append(added,point+0.01)
                added = np.append(added,point-0.01)




            
            hull = ConvexHull(cur_points)

            # Bounding box of the convex hull
            min_x = min(cur_points[:,0])
            min_y = min(cur_points[:,1])
            max_x = max(cur_points[:,0])
            max_y = max(cur_points[:,1])


            bb = [(min_x,min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]

            # plt.subplot(rows,cols,subplots)
            plt.figure(figsize=(3.3,2.7))
            
            
            self.initPlotParams()


            for b in bb:
                plt.plot(b[0], b[1], 'ro',label="Bounding Box" if b==bb[0] else "")

            plt.plot(h_points[:,0], h_points[:,1], 'o',label='Regular Nodes',zorder=1)
            for simplex in hull.simplices:
    
                plt.plot(cur_points[simplex, 0], cur_points[simplex, 1], 'k-')
                
            

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            all_buffers.append(buf.getvalue())
            plt.close()

        while k>0:
            reachable_points = {}
            for point in coords:

                for n in not_connected:

                    if math.dist(point, n) <= router_range :
                        
                        if tuple(point) not in reachable_points:    
                            reachable_points[tuple(point)] = 1

                        else:
                            reachable_points[tuple(point)]+=1

            if len(reachable_points.values()) == 0:
                if len(not_connected) == 0:
                    break
                else:
                    return None
                continue

            max_reachable = max(reachable_points.values())


            best_points = []
            for point in reachable_points:
                if reachable_points[point]==max_reachable:
                    best_points.append(point)

            if len(best_points)>1:
                
                choice = self.getOptimalPoint(possible_loc=best_points,prior1=all_routers,prior2=h_points,range_value=range)
                
                
                if choice ==None:
                    return None
                router_location.append(choice)

                for n in h_points:
                    if math.dist(choice, n) <= router_range:
                        if n in not_connected:
                            not_connected = np.delete(not_connected, np.where(np.all(not_connected == n, axis=1)), axis=0)
                            connected.append(tuple(n))

                
            else:
                choice = best_points[0]
                router_location.append(choice)

                for n in h_points:
                    if math.dist(choice, n) <= router_range:
                        if n in not_connected:
                            not_connected = np.delete(not_connected, np.where(np.all(not_connected == n, axis=1)), axis=0)
                            connected.append(tuple(n))
            all_routers.append(tuple(choice))


            
            k-=1

            plt.figure(figsize=(3.3,2.7))
            
            
            self.initPlotParams()

            plt.plot(central_points[:,0], central_points[:,1], 'bo')


            for best_point in best_points:
                plt.plot(best_point[0], best_point[1], 'x',color='green',zorder=-1)

            for best_point in router_location:
                plt.plot(best_point[0], best_point[1], 'X',color='black',zorder=1)
                circle = plt.Circle((best_point[0], best_point[1]), router_range , color='green', fill = True, linestyle='dotted',alpha=0.1,zorder=1)
                plt.gca().add_patch(circle)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            all_buffers.append(buf.getvalue())

            plt.close()



            

        return router_location


    def placeRouters(self,h_points,coords,range,max_routers,all_routers):
        not_connected = h_points.copy()
        connected= []
        k=max_routers
        router_range = range
        router_location = []

        while k>0:
            reachable_points = {}
            for point in coords:

                for n in not_connected:

                    if math.dist(point, n) <= router_range:
                        
                        if tuple(point) not in reachable_points:    
                            reachable_points[tuple(point)] = 1

                        else:
                            reachable_points[tuple(point)]+=1
                


            if len(reachable_points.values()) == 0:
                if len(not_connected) == 0:
                    break
                continue

            max_reachable = max(reachable_points.values())


            best_points = []
            for point in reachable_points:
                if reachable_points[point]==max_reachable:
                    best_points.append(point)

            if len(best_points)>1:
                
                choice = self.getOptimalPoint(possible_loc=best_points,prior1=all_routers,prior2=h_points,range_value=router_range)
                if choice ==None:
                    return None
                router_location.append(choice)

                for n in h_points:
                    if math.dist(choice, n) <= router_range:
                        if n in not_connected:
                            not_connected = np.delete(not_connected, np.where(np.all(not_connected == n, axis=1)), axis=0)
                            connected.append(tuple(n))

                
            else:
                
                choice = best_points[0]
                
                router_location.append(choice)

                for n in h_points:
                    if math.dist(choice, n) <= router_range:
                        if n in not_connected:
                            not_connected = np.delete(not_connected, np.where(np.all(not_connected == n, axis=1)), axis=0)
                            connected.append(tuple(n))

            all_routers.append(tuple(choice))
            k-=1

            plt.figure(figsize=(3.3,2.7))
            self.initPlotParams()

            plt.plot(h_points[:,0], h_points[:,1], 'bo')

            for best_point in best_points:
                plt.plot(best_point[0], best_point[1], 'x',color='green')
            plt.plot(choice[0], choice[1], 'x',color = 'yellow')

            for best_point in router_location:
                plt.plot(best_point[0], best_point[1], 'X',color='black')
                circle = plt.Circle((best_point[0], best_point[1]), router_range , color='green', fill = True, linestyle='dotted',alpha=0.1)
                plt.gca().add_patch(circle)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            all_buffers.append(buf.getvalue())

            plt.close()
            

        return router_location


    def run(self):

        global all_routers,all_buffers,FIGSIZE


        if not self.initialCheck(self.h_points, self.coords, self.router_range):
            print("Not Possible to connect all the Source points(Not Sufficient Discretization).")
            return None,None
        all_routers = []
        all_buffers = []
        router_range = self.router_range
        max_routers = self.max_routers
        left =  self.h_points.copy()
        H = HeuristicRouterPlacement(self.h_points,self.coords,router_range,max_routers,self.heuristics,self.scale)
        router_location = H.placeRouters(left,self.coords,router_range,max_routers,all_routers)


        in_range = []
        for router1 in router_location:
            for router2 in router_location:
                if math.dist(router1, router2) <= router_range and router1 != router2:
                    in_range.append(router1)


        left = np.array(router_location)

        for router in left:
            if tuple(router) not in all_routers:
                all_routers.append(tuple(router))

        if len(in_range) >0:
            try:
                left = np.delete(left, np.where(np.all(left == np.array(in_range), axis=1)), axis=0)
            except:
                pass
        def are_routers_connected_cpu(routers, r):
                if len(routers) <= 1:
                    return True

                G = netx.Graph()
                G.add_nodes_from(range(len(routers)))
                num_connected = 0
                for i, r1 in enumerate(routers):
                    for j in range(i + 1, len(routers)):
                        if math.dist(r1, routers[j]) <= r:
                            num_connected+=1
                            G.add_edge(i, j)
                    if num_connected == 0:
                        return False

                return netx.is_connected(G)

            

        while not are_routers_connected_cpu(all_routers, router_range):

            router_location = H.addMoreRouters(left,self.coords,router_range,max_routers,all_routers)

            if router_location is None:
                print("Not Possible to connect all the Source points(Not Sufficient Discretization).")
                return None,None


            in_range = []
            for router1 in router_location:
                for router2 in router_location:
                    if math.dist(router1, router2) <= router_range and router1 != router2:
                        in_range.append(router1)


            left = np.array(router_location)


            for router in left:
                if tuple(router) not in all_routers:
                    all_routers.append(tuple(router))

            if len(in_range) >0:
                try:
                    left = np.delete(left, np.where(np.all(left == np.array(in_range), axis=1)), axis=0)
                except:
                    pass

        if self.heuristics[3]:
            all_routers = self.clean(self.h_points,all_routers,router_range)

   
        return all_routers,all_buffers
