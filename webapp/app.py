import flask
from flask import request
from Recommender import BuildRecommender
import random
import networkx as nx
from networkx.readwrite import json_graph


app = flask.Flask(__name__)

rec = BuildRecommender()
races = ['Terran0','Terran1']
tree = nx.DiGraph()
edge_nodes = [1,1]

def race_first_unit(race):
    if race == 'Terran0':
        first_unit = 'CommandCenter0'
    elif race == 'Protoss0':
        first_unit = 'Nexus0'
    elif race =='Zerg0':
        first_unit = 'Hatchery0'
    return first_unit

def default_tree(tree,race):
    first_unit = race_first_unit(race)
    edge_nodes[1] = 1
    tree.add_node(1,name=first_unit,parent='null')
    return tree

@app.route("/")
def index():
    default_race = 'Terran'
    races[0] = default_race + '0'
    races[1] = default_race + '1'
    default_tree(tree,races[0])
    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("index.html",tree_data=tree_data)

@app.route("/default",methods=['GET','POST'])
def default():
    tree.clear()
    races[0] = request.form['friendly_race'] + '0'
    races[1] = request.form['enemy_race'] + '1'
    default_tree(tree,races[0])
    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("index.html",tree_data=tree_data)

@app.route("/recommend",methods=['GET','POST'])
def recommend():   
    if 'autobuild' in request.form.keys():
        autobuild_len = int(request.form["length_autobuild"])
        node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])     
        build_order = [tree.node[nd]['name'] for nd in node_order]
        builds = rec.predict_build(pred_input=build_order,build_length=autobuild_len,races=races)
        for bld in builds:
            next_node = len(node_order) + 1
            tree.add_node(next_node,name=bld,parent=node_order[-1])
            tree.add_edge(node_order[-1],next_node)
            node_order.append(next_node)
            edge_nodes[1] = next_node   
    else:
        expansion_unit = request.args.get('unit_id')
        custom_build = request.args.get('cust_build')
        if custom_build != 'Custom Build':
            builds = custom_build.split(',')
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])
            build_order = [tree.node[nd]['name'] for nd in node_order]
            nd_index = build_order.index(expansion_unit)
            expansion_node = node_order[nd_index]
            tree.node[expansion_node]["name"] = builds[0]
            edge_nodes[1] = expansion_node
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])            
            for nd in tree.nodes():
                if nd not in node_order:
                    tree.remove_node(nd)
            for bld in builds[1:]:
                next_node = len(node_order) + 1
                tree.add_node(next_node,name=bld,parent=node_order[-1])
                tree.add_edge(node_order[-1],next_node)
                node_order.append(next_node)
                edge_nodes[1] = next_node 
        else:
            node_order = nx.shortest_path(tree,source=edge_nodes[0],target=edge_nodes[1])
            build_order = [tree.node[nd]['name'] for nd in node_order]
            next_build = rec.predict_build(pred_input=build_order,build_length=1,races=races)[-1]
            next_node = len(node_order) + 1
            tree.add_node(next_node,name=next_build,parent=node_order[-1])
            tree.add_edge(node_order[-1],next_node)
            edge_nodes[1] = next_node

    tree_data = json_graph.tree_data(tree,root=1)
    return flask.render_template("index.html",tree_data=tree_data)


if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    #os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = False
    app.run()