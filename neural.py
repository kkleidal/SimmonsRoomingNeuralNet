import rooms
import available
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, GaussianLayer, FullConnection
import shelve

brain_shelf = shelve.open("brain.shelf")

grts = []
grt_map = {}
curvy_map = {"None": 0, "Minor": 1, "Major": 2, "Awful": 3}

use_grt = 1
use_size = 1
use_view = 1
use_position_x = 1
use_position_y = 1
use_curvy_wall = 1
use_floor = 1
use_bathroom = 1

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
    if use_grt > 0:
        if room.grt not in grt_map:
            grts.append(room.grt)
            grt_map[room.grt] = len(grts) - 1
        feature.append(grt_map[room.grt] / float(preprocessed[0]))
    if use_size > 0:
        feature.append(int(room.size) / float(preprocessed[1]))
    if use_view > 0:
        feature.append((1 if room.view == "Boston" else 0))
    if use_position_x > 0 or use_position_y > 0:
        feature.append(room.X / float(preprocessed[3][0]))
        feature.append(room.Y / float(preprocessed[3][1]))
    if use_curvy_wall > 0:
        feature.append(curvy_map[room.hasCurvyWall] / float(preprocessed[4]))
    if use_floor > 0:
        feature.append(int(room.num[0]) / float(preprocessed[5]))
    if use_bathroom > 0:
        feature.append(room.get_bathroom() / 2.)
    return feature

labeled_rooms = {"224B": 1, "322D": 1, "421B": 7, "244C": 7, "522C": 1, '740': 1, '631C': 1, '839': 4, '744': 3, '631D': 3, '450': 10, '571': 6, '628': 3, '550': 3, '522C': 1, '977': 2, '976': 6, '975': 8, '636': 1, '1077': 3, '874': 1, '875': 8, '872': 6, '1073': 1, '479B': 2, '731': 2, '540': 0, '1034': 1, '653': 1, '544': 5, '545': 8, '739': 1, '1031B': 5, '1031C': 1, '629': 1, '427': 3, '644': 1, '578A': 8, '578B': 2, '637': 1, '624B': 1, '447': 1, '672': 1, '665': 0, '1025': 2, '372': 0, '1044': 2, '738C': 7, '865': 5, '549A': 2, '374': 2, '378': 1, '643': 3, '477': 1, '474': 2, '649': 0, '373': 1, '376': 4, '925': 3, '538': 5, "972": 9, "936": 8, "865": 4, "1052B": 7, "1065": 4, "448": 10, "421B": 7, "429": 10, "426": 9, "1038": 2}

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

main(T=175, load_brain=True, save_brain=False)
brain_shelf.close()
