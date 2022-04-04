# import heapq to get access to priority queue methods
import heapq
# import math to get access to sqrt function for heuristic
import math

# create action mapping dictionary to determine what actions are possible (easy searching by keys)
actionMap = {
    1:[1,0,0],
    2:[-1,0,0],
    3:[0,1,0],
    4:[0,-1,0],
    5:[0,0,1],
    6:[0,0,-1],
    7:[1,1,0],
    8:[1,-1,0],
    9:[-1,1,0],
    10:[-1,-1,0],
    11:[1,0,1],
    12:[1,0,-1],
    13:[-1,0,1],
    14:[-1,0,-1],
    15:[0,1,1],
    16:[0,1,-1],
    17:[0,-1,1],
    18:[0,-1,-1]
}

# define class to store all the relevant information for the cave problem
class CaveProblem:
    # create CaveProblem constructor
    def __init__(self, type: str, gridSize: tuple, numNodes: int, startingNode: tuple, endingNode: tuple,
                 nodeList: dict):
        self.type = type # type of search: BFS, UCS, A*
        self.gridSize = gridSize # tuple of grid size (x, y, z)
        self.numNodes = numNodes # number of total nodes in the maze
        self.startingNode = startingNode # tuple for starting node
        self.endingNode = endingNode # tuple for ending node
        self.nodeList = nodeList # dictionary containing tuple:list for node:available actions

    # string method
    def __str__(self):
        string = "Type: " + self.type + "\n"
        string += "Num Nodes: " + str(self.numNodes) + "\n"
        string += "Starting Node: " + str(self.startingNode) + "\n"
        string += "Ending Node: " + str(self.endingNode) + "\n"
        string += "Node List:"
        for key in self.nodeList.keys():
            string += "\nNode: " + str(key) + " Actions: " + str(self.nodeList[key])

        return string

# function to read a file and generate a cave problem
def parseFile(fileName: str) -> CaveProblem:
    # open a file for reading
    file = open(fileName, "r")
    # get the type and strip the \n at the end
    searchType = file.readline().strip()
    # read the maze details
    gridSize = tuple((int(i) for i in file.readline().strip().split(" ")))
    startNode = tuple((int(i) for i in file.readline().strip().split(" ")))
    endNode = tuple((int(i) for i in file.readline().strip().split(" ")))
    numNodes = int(file.readline().strip())
    nodeList = dict()
    # start reading in the nodes store (x,y,z) as keys and actions as values
    for i in range(numNodes):
        valList = [int(i) for i in file.readline().strip().split(" ")]
        node = (valList[0], valList[1], valList[2])
        nodeAction = valList[3:]
        nodeList[node] = nodeAction
    # create the cave problem
    problem = CaveProblem(searchType, gridSize, numNodes, startNode, endNode, nodeList)
    # close the file
    file.close()
    return problem

def isValidAction(node: tuple, nodeList: dict, action: int):
    actionList = actionMap.get(action)
    newNode = (node[0]+actionList[0], node[1]+actionList[1], node[2]+actionList[2])
    return newNode in nodeList.keys()

def bfs(start: tuple, end: tuple, nodeList: dict):
    # BFS with uniform edges
    # create the fontier list (list of nodes to be expanded)
    frontierList = list()
    # trace where we have been (a list of visited paths)
    tracePath = dict()

    # add the starting node to the frontier list
    # use a tuple with node and parent node
    frontierList.append((start, None))

    # loop while the frontier list is not empty
    while len(frontierList) > 0:
        # remove the first node/parent combo from the frontier list
        node, parentNode = frontierList.pop(0)

        # if the node is already in the nodes expanded, move on
        if node in tracePath:
            continue
        elif node == end: # if the node is the goal node
            # add goal node to the list of expanded nodes and return the list
            tracePath[node] = (parentNode)
            return tracePath
        else: # node not expanded and not the goal
            # add the node to the path key:node value:parent
            tracePath[node] = (parentNode)
            # expand the node based on actions
            for action in nodeList[node]:
                # create a new node based on the action given
                if isValidAction(node, nodeList, action):
                    newNode = createNodeFromAction(node, action)
                    # if the node created is not in the list of expanded nodes, add it to the frontier list at the end
                    if newNode not in tracePath:
                        frontierList.append((newNode, node))

    # frontier list is empty and goal node not found, no solution possible
    return "FAIL"

def ucs(start: tuple, end: tuple, nodeList: dict):
    frontierList = list()
    tracePath = dict()

    # roughly same algorithm as BFS, but each edge has different value, and the frontier list is a priority queue that
    # sorts based on the first value of the tuple (edge cost)
    heapq.heappush(frontierList, (0, start, None, None))

    while len(frontierList) > 0:
        # pull the path cost, node, parent node, and parent action (action performed to arrive at this node)
        pathCost, node, parent, action = heapq.heappop(frontierList)

        # if the node has already been expanded AND the expanded node has a lower cost, then skip this node
        if node in tracePath and tracePath[node][0] < pathCost:
            continue
        else:
            # node is goal node
            tracePath[node] = (pathCost, parent, action)
            if node == end:
                return tracePath
            # expand the node based on the possible actions
            for act in nodeList[node]:
                # make sure action creates a valid node
                if isValidAction(node, nodeList, act):
                    # create the node and calculate the cost based on the action (10 or 14)
                    newNode = createNodeFromAction(node, act)
                    newCost = pathCost + getCost(act,"UCS")

                    # if the node is not already expanded (because we have been adding more) or if it has been
                    # expanded and the cost of this node is less than the cost of the previously expanded node,
                    # then add to the frontier list
                    if newNode not in tracePath or tracePath[newNode][0] > newCost:
                        heapq.heappush(frontierList,(newCost, newNode, node, act))

    return "FAIL"

def astar(start: tuple, end: tuple, nodeList: dict):
    # This is the exact same algorithm as the UCS except we modify cost to include future cost based on the
    # admissible heuristic

    # The heuristic for determining future cost is that the future cost is the straight-line distance from the
    # current node to the goal node. This heuristic is admissible since the shortest path to the goal from a node is
    # a straight line so that distance is always less than or equal to the actual future cost.

    frontierList = list()
    tracePath = dict()

    heapq.heappush(frontierList, (admissibleHeuristic(start, end), start, None, None))

    while len(frontierList) > 0:
        pathCost, node, parent, action = heapq.heappop(frontierList)

        if node in tracePath and tracePath[node][0] < pathCost:
            continue
        else:
            if node == end:
                tracePath[node] = (pathCost, parent, action)
                return tracePath

            tracePath[node] = (pathCost, parent, action)

            for act in nodeList[node]:
                if isValidAction(node, nodeList, act):
                    newNode = createNodeFromAction(node, act)
                    # calculate total cost based on action and future cost
                    newCost = pathCost + getCost(act, "A*") + admissibleHeuristic(newNode, end)

                    if newNode not in tracePath or tracePath[newNode][0] > newCost:
                        heapq.heappush(frontierList, (newCost, newNode, node, act))
    return "FAIL"

def admissibleHeuristic(node: tuple, end: tuple):
    #an admissible heuristic for future cost is the distance from the current node to the goal node
    #calculated using sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    return math.floor(math.sqrt((end[0]-node[0])**2 + (end[1]-node[1])**2 + (end[2]-node[2])**2))


# Generate a new node from an existing node based on a possible action in the actionMap
def createNodeFromAction(node, action):
    actionList = actionMap.get(action)
    return tuple((node[0]+actionList[0], node[1]+actionList[1], node[2]+actionList[2]))

# Determine action cost based on search type and action type
def getCost(action: int, type: str):
    if type == "BFS":
        # Same cost regardless of action
        return 1
    else:
        # Cost estimated to either 10 or 14. Cost of 10 is a single axis action, cost of 14 is a dual axis action
        if action > 6:
            return 14
        else:
            return 10

# Main function, where the program begins
def main():
    # Open a file to read and get the Cave Problem
    problem = parseFile("input.txt")
    # Open a file to write
    outputFile = open("output.txt", "w")
    # If the end node is not even in the list of nodes, there is no solution to the maze
    if problem.endingNode not in problem.nodeList:
        print("FAIL", file=outputFile)
        return

    # End node is in the list of valid nodes, a solution may exist
    if problem.type == "BFS":
        # run BFS and return a dictionary of node:parent
        result = bfs(problem.startingNode, problem.endingNode, problem.nodeList)
    elif problem.type == "UCS":
        # run UCS and return a dictionary of node:(cost, parent, action)
        result = ucs(problem.startingNode, problem.endingNode, problem.nodeList)
    elif problem.type == "A*":
        # run A* and return a dictionary of node:(cost, parent, action)
        result = astar(problem.startingNode, problem.endingNode, problem.nodeList)
    else:
        # Search type is not BFS, UCS, or A* (probably unnecessary given that input files must be valid)
        result = "FAIL"

    # If search failed (this is for any of the search algorithms)
    if result == "FAIL":
        # Write fail to the file
        print("FAIL", file=outputFile)
    else:
        # Search completed, write to output
        if problem.type == "BFS":
            # BFS uses uniform cost for each edge so we didn't keep track of it in the algorithm
            traceList = list()
            node = problem.endingNode
            while result[node] is not None:
                traceList.append(node)
                node = result[node]
            traceList.append(node)
            traceList.reverse()
            print(len(traceList)-1, file=outputFile)
            print(len(traceList), file=outputFile)
            for index in range(len(traceList)):
                node = traceList[index]
                if index == 0:
                    print(node[0], node[1], node[2], 0, file=outputFile)
                else:
                    print(node[0], node[1], node[2], getCost(0, "BFS"), file=outputFile)
        elif problem.type == "UCS" or problem.type == "A*":
            # UCS and A* use non-uniform edges so we have to sum them up as we go
            traceList = list()
            # Start at the goal node
            node = problem.endingNode
            totalCost = 0
            # Loop till the start node is found
            while result[node][1] is not None:
                # Append the node to the traceList (no need for dictionary since we are not searching the list later)
                # the result tuple has the action at the last element. Get the cost based on the action and the
                # search type
                traceList.append((node, getCost(result[node][2], "UCS")))
                totalCost += getCost(result[node][2],"UCS")
                # the next node is the parent of the current node. Parent node is the second element of the tuple
                node = result[node][1]
            # append on the start node with a cost of 0 since it's the start
            traceList.append((node, 0))
            # reverse the list so we start at start node and end at goal node
            traceList.reverse()
            # print first line of the file which is total path cost of optimal path
            print(totalCost, file=outputFile)
            # print second line of file which is the number of nodes traversed
            print(len(traceList), file=outputFile)
            # print the list of nodes. each node is a tuple with ((x, y, z), cost)
            for tupleItem in traceList:
                node = tupleItem[0]
                cost = tupleItem[1]
                print(node[0], node[1], node[2], cost, file=outputFile)
    # close the file
    outputFile.close()


# call main()
main()
