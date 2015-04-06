# ****************************************
# *                                      *
# *      THE ROOMS OF SIMMONS HALL       *
# *                                      *
# ****************************************

#  by Cosmos Darwin (cosmosd), Sept. 2013


def find(number):
    '''
    returns room object, by room number
    '''
    for room in allRooms:
        if room.num == str(number):
            return room
    return None

def findBathroom(number):
    '''
    returns bathroom object, by bathroom number
    '''
    for bathroom in allBathrooms:
        if bathroom.num == str(number):
            return bathroom
    return None

def dist(r1, r2):
    '''
    returns x and y distance between two rooms, measured in windows
    '''
    room1 = find(r1)
    room2 = find(r2)
    return str(abs(room1.X - room2.X)) + " windows horizontally, " + str(abs(room1.Y - room2.Y)) + " windows vertically (i.e. " + str(abs(room1.Y - room2.Y)/3) + " floors)."

allRooms = []
allSections = []
allBathrooms = []

class Section:
    def __init__(self, label, rooms):
        self.label = label
        self.rooms = rooms

bathroom_type = {"hallway": 0, "insuite": 1, "internal": 2}
class Room:
    def __init__(self, number, section):
        self.num = str(number)             # STRING, e.g. "741A"
        self.grt = str(section)            # STRING, e.g. "23AB"
        self.capacity = None               # STRING, "Single" or "Double"
        self.size = None                   # STRING, e.g. "285" (in square feet)
        self.view = None                   # STRING, "Boston" or "Cambridge"
        self.X = None                      # INT, e.g. 117 (in windows)
        self.Y = None                      # INT, e.g. 12 (in windows)
        self.hasCurvyWall = "None"         # STRING, "None" or "Minor" or "Major" or "Awful"
        self.bathroom = None               # STRING, e.g. "675A" (can be used to lookup bathroom object)

    def get_bathroom(self):
        return bathroom_type[findBathroom(self.bathroom).location]

    def __str__(self):
        return "NUMBER: " + str(self.num) + ", OCCUPANCY: " + self.capacity + ", SIZE: " + str(self.size) + " sq ft, GRT: " + str(self.grt) + ", BATHROOM: " + str(findBathroom(self.bathroom).location) + ", CURVY WALL: " + self.hasCurvyWall + ", VIEW: " + str(self.view) + ", X: " + str(self.X) + " windows"  + ", Y: " + str(self.Y) + " windows"

    def __repr__(self):
        return self.__str__()

class Bathroom:
    def __init__(self, number, location, rooms):
        self.num = str(number)
        self.rooms = rooms
        self.location = location

        # assign self to rooms, self-check
        for roomNumber in self.rooms:
            room = find(roomNumber)
            # if non-existent room
            if room == None:
                print str(roomNumber) + " doesn't exist."
            # if already assigned
            if room.bathroom != None:
                print str(room.num) + " already has a bathroom!"
            room.bathroom = self.num

    def __str__(self):
        return "NUMBER: " + str(self.num) + ", LOCATION: " + self.location + ", ROOMS: " + str(self.rooms)

    def __repr__(self):
        return self.__str__()


# ****************************************
# *         CREATE GRT SECTIONS          *
# ****************************************

# source: curated by Cosmos Darwin '15

allSections.append(Section('23AB', ['224A', '225', '224B', '244B', '244C', '252', '228', '229', '321', '322B', '322C', '324', '326', '341', '344', '322D', '325', '327', '345', '329', '328', '330']))
allSections.append(Section('34C', ['337', '340A', '340B', '379A', '371', '381', '372', '373', '374', '375', '376', '377', '378', '379B', '380', '438', '440', '472', '473', '474', '475', '476', '477', '478', '479A', '479B']))
allSections.append(Section('4AB', ['421C', '422B', '424A', '428', '446', '421B', '422C', '424B', '426', '427', '447', '448', '433', '451', '464', '467', '429', '450', '452', '465', '466']))
allSections.append(Section('5AB', ['521C', '522B', '524', '548', '521B', '522C', '525', '526', '527C', '543', '544', '545', '549A', '549B', '533', '553', '532', '534', '535', '550', '551', '552', '564', '565', '566', '569']))
allSections.append(Section('6AB', ['627', '645B', '624A', '624B', '625', '626', '643', '644', '645A', '647', '648', '649', '650', '652', '664', '628', '629', '631B', '631C', '631D', '632', '633', '653', '665', '667']))
allSections.append(Section('56C', ['537', '575', '573', '574', '577', '578A', '578B', '538', '539', '540', '570', '571', '572A', '572B', '634', '635', '675', '636', '637', '638', '639', '640', '670', '672', '673', '674', '678']))
allSections.append(Section('7ABC', ['721', '724', '741A', '725', '727', '728', '743', '744', '746', '747', '748C', '741B', '729', '732', '733', '730', '731', '748B', '738B', '780', '778', '776', '779', '739', '775', '740', '738C']))
allSections.append(Section('8910A', ['824', '821', '846', '825', '924', '921C', '922C', '921B', '945', '922D', '925', '946', '1024', '1021D', '1022B', '1021C', '1045', '1022C', '1044', '1025', '1043', '1046', '1021A']))
allSections.append(Section('8910B', ['832', '866', '833', '871','865', '872', '932', '971', '931', '933', '966', '972', '936', '1036', '1052A', '1064', '1032', '1066', '1072', '1033', '1052B', '1065', '1031B', '1034', '1031C']))
allSections.append(Section('8910C', ['840', '878', '839', '873', '874', '875', '939', '978', '940', '941', '974', '973', '977', '979', '976', '938', '975', '1078B', '1040', '1039', '1073', '1074', '1076', '1077', '1038', '1078A', '1075']))

# ****************************************
# *             CREATE ROOMS             *
# ****************************************

# source: sections, above

for section in allSections:
    for room in section.rooms:
        allRooms.append(Room(room, section.label))

# ****************************************
# *           CREATE BATHROOMS           *
# ****************************************

# source: curated by Tim Wilczynski '14, Jesse Orlowski '16, Cosmos Darwin '15, and Will Oursler '15

Bathrooms7thFloor = [['721A', 'internal', ['721']],['741C', 'insuite', ['741A', '741B']],['725A', 'hallway', ['724', '725']],['743A', 'hallway', ['743', '744']],['747A', 'hallway', ['746', '747']],['748D', 'insuite', ['748B', '748C']],['727A', 'hallway', ['727']],['728A', 'hallway', ['728', '729']],['730A', 'hallway', ['730', '731']],['732A', 'internal', ['732']],['733A', 'hallway', ['733']],['738A', 'insuite', ['738B', '738C']],['775A', 'hallway', ['775', '776']],['740A', 'hallway', ['739', '740']],['778A', 'hallway', ['778', '779']],['780A', 'internal', ['780']]]
Bathrooms6thFloor = [['644A', 'hallway', ['643', '644']], ['624C', 'insuite', ['624A', '624B']], ['645C', 'insuite', ['645B', '645A']], ['648A', 'hallway', ['647', '648']], ['649A', 'hallway', ['649', '650']], ['626A', 'hallway', ['625', '626']], ['627A', 'hallway', ['627']], ['628A', 'hallway', ['628', '629']], ['652A', 'hallway', ['652', '653']], ['664A', 'internal', ['664']], ['631A', 'hallway', ['631B', '631C', '631D']], ['665A', 'hallway', ['665', '667']], ['633A', 'hallway', ['632','633']], ['634A', 'internal', ['634']], ['635A', 'internal', ['635']], ['670A', 'internal', ['670']], ['674A', 'hallway', ['672','673', '674']], ['636A', 'hallway', ['636','637', '638']], ['675A', 'hallway', ['675']], ['640A', 'hallway', ['639','640']], ['678A', 'internal', ['678']]]
Bathrooms5thFloor = [['522A', 'insuite', ['522B','522C']], ['521A', 'insuite', ['521B','521C']], ['545A', 'hallway', ['543','544', '545']], ['525A', 'hallway', ['524','525']], ['548A', 'hallway', ['548']], ['527A', 'hallway', ['526','527C']], ['549C', 'insuite', ['549B','549A']], ['551A', 'hallway', ['550','551']], ['552A', 'hallway', ['552','553']],['564A', 'internal', ['564']], ['532A', 'internal', ['532']], ['565A', 'hallway', ['565','566']], ['535B', 'hallway', ['534','535']], ['533A', 'hallway', ['533']], ['571A', 'hallway', ['569','570', '571']], ['572C', 'insuite', ['572A','572B']], ['574A', 'hallway', ['573','574']], ['537A', 'hallway', ['537','538']], ['575A', 'internal', ['575']], ['540A', 'hallway', ['539','540']], ['577A', 'hallway', ['577']], ['578C', 'insuite', ['578A','578B']]]
Bathrooms4thFloor = [['421A', 'insuite', ['421B','421C']], ['422A', 'insuite', ['422B','422C']], ['424C', 'insuite', ['424B','424A']], ['427A', 'hallway', ['426','427']], ['446A', 'hallway', ['446']], ['447A', 'hallway', ['447','448']], ['428A', 'hallway', ['428','429']], ['450A', 'internal', ['450']], ['451A', 'hallway', ['451','452']], ['464A', 'internal', ['464']], ['465A', 'hallway', ['465','466']], ['433A', 'internal', ['433']], ['467A', 'internal', ['467']], ['473A', 'hallway', ['473','474', '475']], ['472A', 'internal', ['472']], ['438A', 'internal', ['438']], ['440A', 'internal', ['440']], ['479C', 'insuite', ['479A', '479B']], ['478A', 'hallway', ['476', '477', '478']]]
Bathrooms3rdFloor = [['321A', 'internal', ['321']], ['341A', 'internal', ['341']], ['322A', 'insuite', ['322C','322D']], ['322BA', 'internal', ['322B']], ['325A', 'hallway', ['325','324']], ['345A', 'hallway', ['345','344']], ['326A', 'internal', ['326']], ['327A', 'hallway', ['327']], ['328A', 'internal', ['328']], ['329A', 'hallway', ['329','330']], ['372A', 'hallway', ['371','372']], ['373A', 'hallway', ['373','374','375']], ['337A', 'internal', ['337']], ['378A', 'hallway', ['376','377','378']], ['340AA', 'internal', ['340A']], ['340BA', 'internal', ['340B']], ['379C', 'insuite', ['379B','379A']], ['381A', 'hallway', ['381','380']]]
Bathrooms2ndFloor = [['244A', 'insuite', ['244B','244C']], ['224C', 'insuite', ['224A','224B']], ['228C', 'insuite', ['228','229']], ['252A', 'internal', ['252']], ['225A', 'internal', ['225']]]


Bathrooms8thFloor = [['821A','internal',['821']],['824A','hallway',['824','825']],['846A','hallway',['846']],['832A','internal',['832']],['833A','hallway',['833']],['865A','hallway',['865','866']],['872A','hallway',['871','872']],['873A','hallway',['873','874','875']],['840A','hallway',['839','840']],['878A','internal',['878']]]
Bathrooms9thFloor = [['921BA','internal',['921B']],['921A','insuite',['921C']],['922A','insuite',['922C','922D']],['924A','hallway',['924','925']],['945A','hallway',['945','946']],['931A','internal',['931']],['932A','internal',['932']],['933A','hallway',['933']],['936A','hallway',['936']],['965','hallway',['966']],['971A','hallway',['971','972']],['938A','internal',['938']],['939A','internal',['939']],['940AA','internal',['940']],['941A','internal',['941']],['973A','hallway',['973','974','975']],['977A', 'hallway', ['976','977']],['978A','hallway',['978','979']]]
Bathrooms10thFloor = [['1021B','insuite',['1021A','1021D']],['1021CA','internal',['1021C']],['1022A','insuite',['1022B','1022C']],['1024A','hallway',['1024','1025']],['1044A','hallway',['1043','1044']],['1045A','hallway',['1045','1046']],['1031A','insuite',['1031B','1031C']],['1032A','internal',['1032']],['1033A','hallway',['1033','1034']],['1036A','hallway',['1036']],['1052C','insuite',['1052A','1052B']],['1064A','hallway',['1064','1065']],['1066A','internal',['1066']],['1072A','internal',['1072']],['1038A','internal',['1038']],['1039A','internal',['1039']],['1040AA','internal',['1040']],['1073A','hallway',['1073','1074']],['1077A','hallway',['1075','1076','1077']],['1078C','insuite',['1078A','1078B']]]

BathroomFloors = [Bathrooms10thFloor, Bathrooms9thFloor, Bathrooms8thFloor, Bathrooms7thFloor, Bathrooms6thFloor, Bathrooms5thFloor, Bathrooms4thFloor, Bathrooms3rdFloor, Bathrooms2ndFloor]

for floor in BathroomFloors:
    for bathroom in floor:
        allBathrooms.append(Bathroom(bathroom[0], bathroom[1], bathroom[2]))

# ****************************************
# *            CAPACITY, SIZES           *
# ****************************************

# source: Simmons DB (via dbscaper.py)

listOfCapacitiesAndSizes = [['939', 'Double', '194'], ['340B', 'Double', '204'], ['341', 'Double', '204'], ['440', 'Double', '207'], ['548', 'Double', '207'], ['941', 'Double', '215'], ['824', 'Double', '219'], ['1024', 'Double', '220'], ['322C', 'Double', '220'], ['924', 'Double', '224'], ['446', 'Double', '226'], ['422B', 'Double', '227'], ['921C', 'Double', '228'], ['1036', 'Double', '231'], ['379A', 'Double', '232'], ['522B', 'Double', '233'], ['627', 'Double', '238'], ['1021D', 'Double', '239'], ['1022B', 'Double', '240'], ['421C', 'Double', '240'], ['521C', 'Double', '240'], ['832', 'Double', '241'], ['1078B', 'Double', '243'], ['645B', 'Double', '243'], ['451', 'Double', '244'], ['572B', 'Double', '245'], ['978', 'Double', '245'], ['340A', 'Double', '246'], ['675', 'Double', '246'], ['922C', 'Double', '247'], ['1052A', 'Double', '248'], ['932', 'Double', '248'], ['537', 'Double', '250'], ['738B', 'Double', '250'], ['840', 'Double', '252'], ['533', 'Double', '253'], ['472', 'Double', '255'], ['464', 'Double', '256'], ['553', 'Double', '256'], ['329', 'Double', '257'], ['664', 'Double', '257'], ['780', 'Double', '257'], ['322B', 'Double', '258'], ['344', 'Double', '259'], ['428', 'Double', '259'], ['652', 'Double', '259'], ['252', 'Double', '260'], ['1040', 'Double', '261'], ['721', 'Double', '261'], ['940', 'Double', '264'], ['971', 'Double', '264'], ['224A', 'Double', '265'], ['228', 'Double', '266'], ['424A', 'Double', '267'], ['524', 'Double', '267'], ['225', 'Double', '268'], ['324', 'Double', '268'], ['326', 'Double', '268'], ['433', 'Double', '268'], ['724', 'Double', '268'], ['438', 'Double', '271'], ['650', 'Double', '271'], ['624A', 'Double', '272'], ['821', 'Double', '272'], ['371', 'Double', '273'], ['866', 'Double', '273'], ['321', 'Double', '274'], ['575', 'Double', '274'], ['833', 'Double', '274'], ['778', 'Double', '277'], ['878', 'Double', '277'], ['729', 'Double', '278'], ['1064', 'Double', '282'], ['635', 'Double', '283'], ['1032', 'Double', '285'], ['467', 'Double', '285'], ['732', 'Double', '285'], ['337', 'Double', '287'], ['328', 'Double', '294'], ['846', 'Double', '296'], ['776', 'Double', '297'], ['1066', 'Double', '298'], ['1072', 'Double', '298'], ['1039', 'Double', '299'], ['871', 'Double', '303'], ['381', 'Double', '304'], ['931', 'Double', '304'], ['933', 'Double', '304'], ['741A', 'Double', '312'], ['634', 'Double', '313'], ['966', 'Double', '313'], ['921B', 'Double', '320'], ['1021C', 'Double', '326'], ['945', 'Double', '329'], ['733', 'Double', '346'], ['452', 'Single', '117'], ['653', 'Single', '117'], ['779', 'Single', '119'], ['427', 'Single', '120'], ['527C', 'Single', '121'], ['540', 'Single', '121'], ['626', 'Single', '121'], ['372', 'Single', '122'], ['665', 'Single', '122'], ['379B', 'Single', '123'], ['447', 'Single', '123'], ['473', 'Single', '123'], ['549B', 'Single', '123'], ['631B', 'Single', '123'], ['728', 'Single', '123'], ['731', 'Single', '123'], ['973', 'Single', '123'], ['322D', 'Single', '124'], ['325', 'Single', '124'], ['373', 'Single', '124'], ['378', 'Single', '124'], ['465', 'Single', '124'], ['479B', 'Single', '124'], ['525', 'Single', '124'], ['624B', 'Single', '124'], ['740', 'Single', '124'], ['1031C', 'Single', '125'], ['1045', 'Single', '125'], ['1073', 'Single', '125'], ['552', 'Single', '125'], ['636', 'Single', '125'], ['640', 'Single', '125'], ['648', 'Single', '125'], ['725', 'Single', '125'], ['727', 'Single', '125'], ['865', 'Single', '125'], ['1022C', 'Single', '126'], ['1044', 'Single', '126'], ['224B', 'Single', '126'], ['422C', 'Single', '126'], ['424B', 'Single', '126'], ['466', 'Single', '126'], ['551', 'Single', '126'], ['574', 'Single', '126'], ['577', 'Single', '126'], ['633', 'Single', '126'], ['649', 'Single', '126'], ['730', 'Single', '126'], ['743', 'Single', '126'], ['748C', 'Single', '126'], ['478', 'Single', '127'], ['578B', 'Single', '127'], ['647', 'Single', '127'], ['775', 'Single', '127'], ['873', 'Single', '127'], ['565', 'Single', '128'], ['739', 'Single', '128'], ['747', 'Single', '128'], ['922D', 'Single', '128'], ['1031B', 'Single', '129'], ['421B', 'Single', '129'], ['522C', 'Single', '129'], ['872', 'Single', '129'], ['1034', 'Single', '130'], ['1052B', 'Single', '130'], ['229', 'Single', '130'], ['539', 'Single', '130'], ['1065', 'Single', '131'], ['345', 'Single', '131'], ['629', 'Single', '131'], ['534', 'Single', '132'], ['639', 'Single', '132'], ['645A', 'Single', '132'], ['672', 'Single', '132'], ['521B', 'Single', '133'], ['572A', 'Single', '133'], ['1074', 'Single', '134'], ['549A', 'Single', '134'], ['974', 'Single', '134'], ['550', 'Single', '135'], ['631C', 'Single', '135'], ['874', 'Single', '135'], ['1033', 'Single', '136'], ['448', 'Single', '136'], ['573', 'Single', '136'], ['632', 'Single', '136'], ['674', 'Single', '136'], ['644', 'Single', '137'], ['673', 'Single', '137'], ['377', 'Single', '138'], ['477', 'Single', '138'], ['570', 'Single', '138'], ['571', 'Single', '138'], ['637', 'Single', '138'], ['744', 'Single', '138'], ['1076', 'Single', '139'], ['475', 'Single', '139'], ['566', 'Single', '139'], ['746', 'Single', '139'], ['569', 'Single', '140'], ['1077', 'Single', '141'], ['474', 'Single', '141'], ['748B', 'Single', '141'], ['977', 'Single', '142'], ['374', 'Single', '143'], ['544', 'Single', '145'], ['380', 'Single', '147'], ['643', 'Single', '147'], ['825', 'Single', '147'], ['670', 'Single', '148'], ['972', 'Single', '148'], ['532', 'Single', '150'], ['925', 'Single', '150'], ['376', 'Single', '152'], ['741B', 'Single', '153'], ['1025', 'Single', '154'], ['578A', 'Single', '154'], ['375', 'Single', '155'], ['625', 'Single', '155'], ['564', 'Single', '156'], ['678', 'Single', '156'], ['1038', 'Single', '157'], ['426', 'Single', '157'], ['526', 'Single', '157'], ['543', 'Single', '160'], ['738C', 'Single', '160'], ['244B', 'Single', '161'], ['538', 'Single', '161'], ['979', 'Single', '162'], ['479A', 'Single', '163'], ['839', 'Single', '163'], ['1078A', 'Single', '164'], ['976', 'Single', '166'], ['545', 'Single', '170'], ['936', 'Single', '174'], ['938', 'Single', '175'], ['450', 'Single', '180'], ['638', 'Single', '181'], ['667', 'Single', '183'], ['429', 'Single', '186'], ['1043', 'Single', '187'], ['1046', 'Single', '190'], ['1075', 'Single', '190'], ['946', 'Single', '191'], ['330', 'Single', '192'], ['975', 'Single', '193'], ['631D', 'Single', '197'], ['1021A', 'Single', '199'], ['476', 'Single', '208'], ['628', 'Single', '212'], ['327', 'Single', '216'], ['244C', 'Single', '222'], ['535', 'Single', '239'], ['875', 'Single', '253']]
# elements have the form [number, capacity, size]

for entry in listOfCapacitiesAndSizes:
    room = find(str(entry[0]))
    room.capacity = entry[1]
    room.size = entry[2]

# ****************************************
# *         BOSTON/CAMBRIDGE VIEW        *
# ****************************************

# source: 'jQueries' on Map/Directory SVG

listOfBostonSideRooms = ['244C', '244B', '252', '321', '341', '344', '345', '371', '372', '373', '374', '375', '376', '377', '378', '379B', '379A', '381', '380', '472', '473', '474', '475', '476', '477', '478', '479B', '479A', '421B', '421C', '446', '447', '448', '450', '451', '452', '464', '465', '466', '467', '543', '544', '545', '548', '549B', '549A', '550', '551', '552', '553', '564', '565', '566', '569', '570', '571', '572B', '572A', '573', '574', '575', '577', '578B', '578A', '643', '644', '645B', '645A', '647', '648', '649', '650', '652', '653', '664', '665', '667', '670', '672', '673', '674', '675', '678', '775', '776', '778', '779', '780', '741A', '741B', '743', '744', '746', '747', '748C', '748B', '873', '874', '875', '878', '865', '866', '871', '872', '846', '973', '974', '975', '976', '977', '978', '979', '972', '971', '966', '921B', '946', '945', '1073', '1074', '1075', '1076', '1077', '1078B', '1078A', '1072', '1066', '1065', '1064', '1052A', '1052B', '1021C', '1021A', '1043', '1044', '1045', '1046']
listOfCambridgeSideRooms = ['322D', '322C', '322B', '324', '325', '326', '327', '328', '329', '330', '337', '340A', '340B', '440', '438', '422B', '422C', '424A', '424B', '426', '427', '428', '429', '433', '224A', '224B', '225', '228', '229', '521B', '521C', '522B', '522C', '524', '525', '526', '527C', '532', '533', '534', '535', '537', '538', '539', '540', '624A', '624B', '625', '626', '627', '628', '629', '631B', '631C', '631D', '632', '633', '634', '635', '636', '637', '638', '639', '640', '738B', '738C', '739', '740', '733', '732', '731', '730', '729', '728', '727', '725', '724', '721', '840', '839', '832', '833', '821', '824', '825', '938', '939', '940', '941', '931', '932', '933', '936', '921C', '922C', '922D', '924', '925', '1038', '1039', '1040', '1021D', '1022C', '1022B', '1024', '1025', '1031B', '1031C', '1032', '1033', '1034', '1036']

for room in allRooms:
    if room.num in listOfBostonSideRooms:
        room.view = 'Boston'
    elif room.num in listOfCambridgeSideRooms:
        room.view = 'Cambridge'

# ****************************************
# *            X & Y POSITION            *
# ****************************************

# source: 'jQueries' on Map/Directory SVG

listOfTopsAndLefts = [['r244C','1132','814'],['r244B','1132','984'],['r252','1132','1733.2650146484375'],['r321','1030','440'],['r341','1030','576'],['r344','1030','780'],['r345','1030','916'],['r371','1030','3126'],['r372','1030','3228'],['r373','1030','3330'],['r374','1030','3432'],['r375','1030','3534'],['r376','1030','3636'],['r377','1030','3738'],['r378','1030','3840'],['r379B','1030','3942'],['r379A','1030','4044'],['r381','1030','4214'],['r380','1030','4146'],['r472','928','3126'],['r473','928','3330'],['r474','928','3432'],['r475','928','3534'],['r476','928','3636'],['r477','928','3738'],['r478','928','3840'],['r479B','928','3942'],['r479A','928','4044'],['r421B','928','440'],['r421C','928','440'],['r446','928','1228.2919921875'],['r447','928','1494'],['r448','928','1596'],['r450','928','1698'],['r451','928','1800'],['r452','928','2004'],['r464','928','2106'],['r465','928','2310'],['r466','928','2412'],['r467','928','2514'],['r543','826','780'],['r544','826','882'],['r545','826','984'],['r548','826','1292.4940185546875'],['r549B','826','1494'],['r549A','826','1596'],['r550','826','1698'],['r551','826','1800'],['r552','826','1902'],['r553','826','2004'],['r564','826','2208'],['r565','826','2310'],['r566','826','2412'],['r569','826','2514'],['r570','826','2616'],['r571','826','2718'],['r572B','826','2820'],['r572A','826','3024'],['r573','826','3126'],['r574','826','3228'],['r575','826','3330'],['r577','826','3840'],['r578B','826','3942'],['r578A','826','4044'],['r643','724','780'],['r644','724','882'],['r645B','724','984'],['r645A','724','1188'],['r647','724','1290'],['r648','724','1392'],['r649','724','1494'],['r650','724','1596'],['r652','724','1800'],['r653','724','2004'],['r664','724','2106'],['r665','724','2310'],['r667','724','2412'],['r670','724','2820'],['r672','724','2922'],['r673','724','3024'],['r674','724','3126'],['r675','724','3228'],['r678','724','3840'],['r775','622','3330'],['r776','622','3432'],['r778','622','3840'],['r779','622','4146'],['r780','622','4248'],['r741A','622','440'],['r741B','622','440'],['r743','622','1086'],['r744','622','1188'],['r746','622','1290'],['r747','622','1392'],['r748C','622','1494'],['r748B','622','1596'],['r873','520','3330'],['r874','520','3432'],['r875','520','3534'],['r878','520','3840'],['r865','520','2310'],['r866','520','2412'],['r871','520','2620.39111328125'],['r872','520','2820'],['r846','520','1120'],['r973','418','3330'],['r974','418','3432'],['r975','418','3534'],['r976','418','3636'],['r977','418','3738'],['r978','418','3840'],['r979','418','4044'],['r972','418','2820'],['r971','418','2619.6689453125'],['r966','418','2412'],['r921B','418','440'],['r946','418','1188'],['r945','418','949.9340209960938'],['r1073','316','3330'],['r1074','316','3432'],['r1075','316','3534'],['r1076','316','3636'],['r1077','316','3738'],['r1078B','316','3840'],['r1078A','316','4044'],['r1072','316','2718'],['r1066','316','2514'],['r1065','316','2412'],['r1064','316','2208'],['r1052A','316','2004'],['r1052B','316','1902'],['r1021C','316','440'],['r1021A','316','678'],['r1043','316','827.5780029296875'],['r1044','316','984'],['r1045','316','1086'],['r1046','316','1188'],['r322D','1030','678'],['r322C','1030','440'],['r322B','1030','440'],['r324','1030','882'],['r325','1030','1086'],['r326','1030','1290'],['r327','1030','1494'],['r328','1030','1664.7860107421875'],['r329','1030','1906'],['r330','1030','2110'],['r337','1030','3432'],['r340A','1030','3811.71923828125'],['r340B','1030','4146'],['r440','928','4146'],['r438','928','3432'],['r422B','928','440'],['r422C','928','678'],['r424A','928','882'],['r424B','928','1086'],['r426','928','1290'],['r427','928','1392'],['r428','928','1494'],['r429','928','1698'],['r433','928','2412'],['r224A','1132','882'],['r224B','1132','1086'],['r225','1132','1290'],['r228','1132','1735.2650146484375'],['r229','1132','1902'],['r521B','826','440'],['r521C','826','440'],['r522B','826','440'],['r522C','826','678'],['r524','826','882'],['r525','826','1086'],['r526','826','1290'],['r527C','826','1392'],['r532','826','2310'],['r533','826','2412'],['r534','826','2616'],['r535','826','2718'],['r537','826','3432'],['r538','826','3636'],['r539','826','3738'],['r540','826','3840'],['r624A','724','882'],['r624B','724','1086'],['r625','724','1290'],['r626','724','1392'],['r627','724','1494'],['r628','724','1803.6190185546875'],['r629','724','2004'],['r631B','724','2310'],['r631C','724','2412'],['r631D','724','2514'],['r632','724','2616'],['r633','724','2718'],['r634','724','2922'],['r635','724','3126'],['r636','724','3330'],['r637','724','3432'],['r638','724','3534'],['r639','724','3636'],['r640','724','3738'],['r738B','622','3432'],['r738C','622','3636'],['r739','622','3738'],['r740','622','3840'],['r733','622','2412'],['r732','622','2208'],['r731','622','2004'],['r730','622','1902'],['r729','622','1596'],['r728','622','1494'],['r727','622','1392'],['r725','622','1086'],['r724','622','882'],['r721','622','440'],['r840','520','3840'],['r839','520','3707.182861328125'],['r832','520','2208'],['r833','520','2412'],['r821','520','440'],['r824','520','882'],['r825','520','1086'],['r938','418','3432'],['r939','418','3579.178955078125'],['r940','418','3806'],['r941','418','4146'],['r931','418','1902'],['r932','418','2208'],['r933','418','2412'],['r936','418','2639.77685546875'],['r921C','418','440'],['r922C','418','440'],['r922D','418','678'],['r924','418','882'],['r925','418','1086'],['r1038','316','3432'],['r1039','316','3566.715087890625'],['r1040','316','3806'],['r1021D','316','442'],['r1022C','316','678'],['r1022B','316','440'],['r1024','316','882'],['r1025','316','1086'],['r1031B','316','1902'],['r1031C','316','2004'],['r1032','316','2208'],['r1033','316','2412'],['r1034','316','2514'],['r1036','316.0010070800781','2629.59716796875']]
# elements have the form [number, y dist from top, x dist from left]
# all y's are int multiples of 3, x's can be floats because of curvy walls

dimensionOfWindow = 34
for room in listOfTopsAndLefts:
    '''
    yields position of bottom-left corner of room
    relative to west-end at ground level, in windows.
    '''
    room[0] = room[0][1:len(room[0])]
    room[1] = 27 - int((float(room[1]) - 316.0) / dimensionOfWindow) # tops
    room[2] = int((float(room[2]) - 440.0) / dimensionOfWindow) # lefts

for entry in listOfTopsAndLefts:
    room = find(str(entry[0]))
    room.Y = entry[1]
    room.X = entry[2]


# ****************************************
# *            HAS CURVY WALL            *
# ****************************************

# source: curated by Tim Wilczynski '14

listOfMinorCurvyWalls = ['631D', '252', '340A', '438', '545', '627', '875', '839', '866', '871', '945', '966', '971', '938', '1034', '1066']
listOfMajorCurvyWalls = ['1021A','933','833','776','667','337','327','228A']
listOfAwfulCurvyWalls = ['1043','1036','1039','939','936','733','729','628','575','446','548','328']

for room in allRooms:
    if room.num in listOfMinorCurvyWalls:
        room.hasCurvyWall = "Minor"
    if room.num in listOfMajorCurvyWalls:
        room.hasCurvyWall = "Major"
    if room.num in listOfAwfulCurvyWalls:
        room.hasCurvyWall = "Awful"

'''
# ****************************************
# *                                      *
# *       ACCURANCY CHECKS & STATS       *
# *                                      *
# ****************************************


print """
**************************************
*   TOTAL ROOMS & CAPACITY CHECKS    *
**************************************
"""

print "There are " + str(len(allRooms)) + " total rooms."

numberOfDoubles = 0
numberOfSingles = 0
for room in allRooms:
    if room.capacity == 'Double':
        numberOfDoubles += 1
    if room.capacity == 'Single':
        numberOfSingles += 1
print "There are " + str(numberOfDoubles) + " doubles and " + str(numberOfSingles) + " singles."
print "This adds up to " + str((2 * numberOfDoubles) + numberOfSingles) + " residents."

print """
**************************************
*         GRT SECTION CHECKS         *
**************************************
"""

totalResidentsAccountFor = 0
for section in allSections:
    numberOfSinglesInSection = 0
    numberOfDoublesInSection = 0
    for entry in section.rooms:
        room = find(entry)
        if str(room.capacity) == 'Single':
            numberOfSinglesInSection += 1
        elif str(room.capacity) == 'Double':
            numberOfDoublesInSection += 1
    totalResidentsAccountFor += numberOfSinglesInSection + (2 * numberOfDoublesInSection)
    print "GRT Section " + str(section.label).rjust(5) + " has " + str(len(section.rooms)) + " rooms (" + "%02d" % (numberOfSinglesInSection,) + " singles and " + "%02d" % (numberOfSinglesInSection,) + " doubles = " + str(numberOfSinglesInSection + (2 * numberOfDoublesInSection)) + " residents)."
print str(totalResidentsAccountFor) + " residents are assigned to sections."

print """
**************************************
*       SQUARE FOOTAGE CHECKS        *
**************************************
"""

numberOfSizelessRooms = 0
for room in allRooms:
    if str(room.size) == None:
        numberOfSizelessRooms += 1
print str(numberOfSizelessRooms) + " rooms are missing square footage data."

sizeOfLargestSingle = 0
sizeOfSmallestSingle = 1000
sizeOfLargestDouble = 0
sizeOfSmallestDouble = 1000

totalAreaOfSingles = 0
totalAreaOfDoubles = 0

for room in allRooms:
    if str(room.capacity) == 'Single':
        totalAreaOfSingles += int(room.size)
        sizeOfLargestSingle = max(int(room.size), sizeOfLargestSingle)
        sizeOfSmallestSingle = min(int(room.size), sizeOfSmallestSingle)  
    elif str(room.capacity) == 'Double':
        totalAreaOfDoubles += int(room.size)
        sizeOfLargestDouble = max(int(room.size), sizeOfLargestDouble)
        sizeOfSmallestDouble = min(int(room.size), sizeOfSmallestDouble)

averageSizeOfSingle = totalAreaOfSingles / numberOfSingles
averageSizeOfDouble = totalAreaOfDoubles / numberOfDoubles

print "Singles range in size from " + str(sizeOfSmallestSingle) + " sq ft to " + str(sizeOfLargestSingle) + " sq ft, averaging " + str(averageSizeOfSingle) + " sq ft."
print "Doubles range in size from " + str(sizeOfSmallestDouble) + " sq ft to " + str(sizeOfLargestDouble) + " sq ft, averaging " + str(averageSizeOfDouble) + " sq ft."

print """
**************************************
*    BOSTON/CAMBRIDGE SIDE CHECKS    *
**************************************
"""

numberOfViewlessRooms = 0
numberOfCambridgeSideRooms = 0
numberOfBostonSideRooms = 0

for room in allRooms:
    if room.view == 'Boston':
        numberOfBostonSideRooms += 1
    elif room.view == 'Cambridge':
        numberOfCambridgeSideRooms += 1
    elif room.view == None:
        numberOfViewlessRooms += 1

print str(numberOfViewlessRooms) + " rooms are missing view data."
print str(numberOfBostonSideRooms) + " rooms face Boston, " + str(numberOfCambridgeSideRooms) + " rooms face Cambridge."

print """
**************************************
*           POSITION CHECKS          *
**************************************
"""

numberOfRoomsWithNoPositionData = 0
for room in allRooms:
    if str(room.X) == None or str(room.Y) == None:
        numberOfRoomsWithNoPositionData += 1
        print str(room.num) + " is missing position data."
print str(numberOfRoomsWithNoPositionData) + " rooms are missing position data."

for room in allRooms:
    if room.Y % 3 != 0:
        print room.num + " has incorrect vertical position."


print """
**************************************
*        HAS CURVY WALL CHECKS       *
**************************************
"""

print str(len(listOfMinorCurvyWalls) + len(listOfMajorCurvyWalls) + len(listOfAwfulCurvyWalls)) + " rooms have curvy walls, " + str(len(listOfMinorCurvyWalls)) + " minor, " + str(len(listOfMajorCurvyWalls)) + " major, " + str(len(listOfAwfulCurvyWalls)) + " awful."


print """
**************************************
*           BATHROOM CHECKS          *
**************************************
"""

for room in allRooms:
    if room.bathroom == None:
        print room.num + " has no bathroom."

numberOfResidentsWithBathrooms = 0
numberOfRoomsWithBathrooms = 0
for room in allRooms:
    if room.bathroom != None:
        numberOfRoomsWithBathrooms += 1
        if room.capacity == 'Single':
            numberOfResidentsWithBathrooms += 1
        elif room.capacity == 'Double':
            numberOfResidentsWithBathrooms += 2
        else:
            raise "Hell!"

# print "There are " + str(len(allBathrooms)) + " total bathrooms, with an average of " + str( numberOfResidentsWithBathrooms / float(len(allBathrooms)) ) +  " users each." 

numInternal = 0
numInsuite = 0
numHallway = 0
for bathroom in allBathrooms:
    if bathroom.location == 'internal':
        numInternal += 1
    if bathroom.location == 'insuite':
        numInsuite += 1
    if bathroom.location == 'hallway':
        numHallway += 1

# print "There are " + str(numInternal) + " internal bathrooms, " + str(numInsuite) + " insuite bathrooms, and " + str(numHallway) + " hallway bathrooms."

# print str(numberOfRoomsWithBathrooms) + " rooms (" + str(numberOfResidentsWithBathrooms) + " residents)" + " have assigned bathrooms."

numberOfSinglesWithExclusiveBathrooms = 0
numberOfDoublesWithExclusiveBathrooms = 0
numberOfSinglesWithInternalBathrooms = 0
numberOfDoublesWithInternalBathrooms = 0
for room in allRooms:
    if room.capacity == 'Single':
        bath = findBathroom(room.bathroom)
        sharedWith = bath.rooms
        if bath.location == 'internal':
            numberOfSinglesWithInternalBathrooms += 1
            numberOfSinglesWithExclusiveBathrooms += 1
        elif len(sharedWith) == 1:
            numberOfSinglesWithExclusiveBathrooms += 1
    if room.capacity == 'Double':
        bath = findBathroom(room.bathroom)
        sharedWith = bath.rooms
        if bath.location == 'internal':
            numberOfDoublesWithInternalBathrooms += 1
            numberOfDoublesWithExclusiveBathrooms += 1
        elif len(sharedWith) == 1:
            numberOfDoublesWithExclusiveBathrooms += 1

# print "There are " + str(numberOfSinglesWithExclusiveBathrooms) + " singles with their own bathoom, of which " + str(numberOfSinglesWithInternalBathrooms) + " are internal."
# print "There are " + str(numberOfDoublesWithExclusiveBathrooms) + " doubles with their own bathoom, of which " + str(numberOfDoublesWithInternalBathrooms) + " are internal."

# print "Checks completed."
'''
