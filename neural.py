import rooms
import available
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, GaussianLayer, FullConnection
import shelve
import json

brain_shelf = shelve.open("brain.shelf")

grts = []
grt_map = {}
curvy_map = {"None": 0, "Minor": 1, "Major": 2, "Awful": 3}

use_grt = True
use_size = True
use_view = True
use_position_x = True
use_position_y = True
use_curvy_wall = True
use_floor = True
use_bathroom = True

def preprocess_rooms(rooms):
    max_size = None
    max_x_pos = None
    max_y_pos = None
    for room in rooms:
        if room.grt not in grt_map:
            grts.append(room.grt)
            grt_map[room.grt] = len(grts) - 1
        if max_size == None or int(room.size) > max_size:
            max_size = int(room.size)
        if max_x_pos == None or room.X > max_x_pos:
            max_x_pos = room.X
        if max_y_pos == None or room.Y > max_y_pos:
            max_y_pos = room.Y
    return (len(grts), max_size, 1, (max_x_pos, max_y_pos), len(curvy_map), 10)

def room_to_feature_vector(room, preprocessed, use_grt=True, use_size=True, use_view=True, use_position=True, use_curvy_wall=True, use_floor=True):
    feature = []
    if use_grt:
        if room.grt not in grt_map:
            grts.append(room.grt)
            grt_map[room.grt] = len(grts) - 1
        feature.append(grt_map[room.grt] / float(preprocessed[0]))
    if use_size:
        feature.append(int(room.size) / float(preprocessed[1]))
    if use_view > 0:
        feature.append((1 if room.view == "Boston" else 0))
    if use_position_x:
        feature.append(room.X / float(preprocessed[3][0]))
    if use_position_y:
        feature.append(room.Y / float(preprocessed[3][1]))
    if use_curvy_wall:
        feature.append(curvy_map[room.hasCurvyWall] / float(preprocessed[4]))
    if use_floor:
        feature.append(int(room.num[0]) / float(preprocessed[5]))
    if use_bathroom:
        feature.append(room.get_bathroom() / 2.)
    return feature

def getRoomsMap(singles, vectors):
    roomsMap = {}
    for i, room in enumerate(singles):
        roomsMap[room.num] = (room, vectors[i])
    return roomsMap

def getLabeledRoomsFeaturesAndLabels(roomsMap):
    return [(roomsMap[number][0], roomsMap[number][1], labeled_rooms[number]) for number in labeled_rooms]

def main(T=10, load_brain=False, save_brain=False):
    singles = [room for room in rooms.allRooms if room.capacity == "Single"]
    preprocessed = preprocess_rooms(singles)
    all_vectors = [room_to_feature_vector(room, preprocessed) for room in singles]
    
    training_sequences = getLabeledRoomsFeaturesAndLabels(getRoomsMap(singles, all_vectors))
    
    input_units = len(all_vectors[0])

    if load_brain and "net" in brain_shelf:
        net = brain_shelf["net"]
        net.sorted = False
        net.sortModules()
    else:
        net = FeedForwardNetwork() # buildNetwork(len(all_vectors[0]), 10000, 1)
        layer_in = LinearLayer(input_units)
        layer_hidden1 = SigmoidLayer(1000)
        layer_hidden2 = LinearLayer(100)
        layer_out = LinearLayer(1)
        net.addInputModule(layer_in)
        net.addModule(layer_hidden1)
        net.addModule(layer_hidden2)
        net.addOutputModule(layer_out)

        in_to_hidden1 = FullConnection(layer_in, layer_hidden1)
        hidden1_to_hidden2 = FullConnection(layer_hidden2, layer_hidden1)
        hidden1_to_out = FullConnection(layer_hidden1, layer_out)
        hidden2_to_out = FullConnection(layer_hidden1, layer_out)
        net.addConnection(in_to_hidden1)
        net.addConnection(hidden1_to_hidden2)
        net.addConnection(hidden2_to_out)
        # net.addConnection(hidden1_to_out)

        net.sortModules()

        training_data = SupervisedDataSet(len(all_vectors[0]), 1)
        for training_seq in training_sequences: 
            training_data.appendLinked(training_seq[1], training_seq[2])
        trainer = BackpropTrainer(net, training_data)
        for i in xrange(T):
            error = trainer.train()
            print "Training iteration %d.  Error: %f" % (i + 1, error)

        if save_brain:
            brain_shelf["net"] = net
    
    labeled_rooms = []
    for i, vector in enumerate(all_vectors):
        labeled_rooms.append((singles[i], net.activate(vector)))
    
    available_rooms = available.get_available_rooms()

    labeled_rooms.sort(key=lambda x: -x[1])
    for room, label in labeled_rooms:
        if room.num in available_rooms:
            print "%16.12f: %s" % (label, room)

with open('preferences.json', 'r') as f:
    labeled_rooms = json.load(f)

main(T=175, load_brain=True, save_brain=False)
brain_shelf.close()
