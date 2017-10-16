import flask
from flask import request
#from Recommender import Recommender
import random
import networkx as nx
from networkx.readwrite import json_graph


app = flask.Flask(__name__)

#rec = Recommender()
races = ['Terran0','Terran1']
friendly_tree = nx.DiGraph()
enemy_tree = nx.DiGraph()
build_order = []

def default_tree(tree,race):
    if race == 'Terran0':
        first_unit = 'CommandCenter0'
    elif race == 'Protoss0':
        first_unit = 'Nexus0'
    elif race =='Zerg0':
        first_unit = 'Hatchery0'
    build_order.append(first_unit)
    tree.add_node(race,name=race,parent='null')
    tree.add_node(first_unit,name=first_unit,parent=race)
    tree.add_node('custom000',name='custom000',parent=race)
    tree.add_edge(race,first_unit)
    tree.add_edge(race,'custom000')
    return tree

@app.route("/")
def index():
    default_race = 'Terran'
    races[0] = default_race + '0'
    races[1] = default_race + '1'
    default_tree(friendly_tree,races[0])
    enemy_tree.add_node(races[1],name=races[1],parent='null')
    tree_data = [json_graph.tree_data(friendly_tree,root=races[0]),
                 json_graph.tree_data(enemy_tree,root=races[1])
                ]
    return flask.render_template("index.html",tree_data=tree_data)

@app.route("/default",methods=['GET','POST'])
def default():
    friendly_tree.clear()
    enemy_tree.clear()
    races[0] = request.form['friendly_race'] + '0'
    races[1] = request.form['enemy_race'] + '1'
    default_tree(friendly_tree,races[0])
    enemy_tree.add_node(races[1],name=races[1],parent='null')
    tree_data = [json_graph.tree_data(friendly_tree,root=races[0]),
                 json_graph.tree_data(enemy_tree,root=races[1])
                ]
    return flask.render_template("index.html",tree_data=tree_data)

@app.route("/recommend",methods=['GET','POST'])
def recommend():
    expansion_unit = request.args.get('unit_id')
    player = expansion_unit[-1]
    custom = 'custom' in expansion_unit
    if custom:
        unit = request.args.get('cust_build') + player
    else:
        unit = expansion_unit
    build_order.append(unit)
    if player == "0":
        friendly_tree.node[expansion_unit]["name"] = unit
    elif player == "1":
        enemy_tree.node[expansion_unit]["name"] = unit
    else:
        print("Player not found")

    print(json_graph.tree_data(friendly_tree,root=races[0]))

    tree_data = [json_graph.tree_data(friendly_tree,root=races[0]),
                 json_graph.tree_data(enemy_tree,root=races[1])
                ]
    return flask.render_template("index.html",tree_data=tree_data)


if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    #os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = False
    app.run()