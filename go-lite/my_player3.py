import random
#import time
# Game constants
# Board size = 5x5
# Input file name = input.txt
# Output file name = output.txt
BOARD_SIZE = 5
INPUT_FILE_NAME = "input.txt"
OUTPUT_FILE_NAME = "output.txt"
PLAYER_COLOR = 0


# Read board and previous board from input file
def readFile(path=INPUT_FILE_NAME):
    global PLAYER_COLOR
    file = open(path, "r")
    PLAYER_COLOR = int(file.readline().strip())
    prevBoard = list()
    currBoard = list()
    for ind in range(2):
        for ind2 in range(5):
            line = file.readline().strip()
            boardRow = list()
            for item in line:
                boardRow.append(int(item))
            if ind == 0:
                prevBoard.append(boardRow)
            else:
                currBoard.append(boardRow)
    file.close()
    return prevBoard, currBoard


# Create copy of board used to predict future moves
def copyBoard(board):
    copy = list()
    for rowInd in range(BOARD_SIZE):
        rowList = list()
        for colInd in range(BOARD_SIZE):
            rowList.append(board[rowInd][colInd])
        copy.append(rowList)

    return copy


# Given a point on the board, find the neighbors. Used to determine liberties
def findNeighbors(point):
    row = point[0]
    col = point[1]
    possibleAdjacent = [(row+1, col),(row-1, col),(row, col+1),(row, col-1)]
    actualAdjacent = [point for point in possibleAdjacent if BOARD_SIZE > point[0] >= 0 and BOARD_SIZE > point[1] >= 0]

    return actualAdjacent


# Given the board and a point on the board containing a player piece, find all the adjacent neighbors that are the
# same color as the piece. Used to determine liberty for clusters
def findFriendlyNeighbors(board, point):
    allPoints = findNeighbors(point)
    row,col = point[0], point[1]
    color = board[row][col]
    friendlies = list()
    for neighbor in allPoints:
        if board[neighbor[0]][neighbor[1]] == color:
            friendlies.append(neighbor)

    return friendlies


# Use a BFS search to find all the friendly stones touching a stone at a point p. Used to determine a cluster's
# liberties
def findFriendlyCluster(board, point):
    frontier = [point]
    cluster = list()

    while len(frontier) > 0:
        node = frontier.pop(0)
        cluster.append(node)
        friendlyNeighbors = findFriendlyNeighbors(board, node)
        for neighbor in friendlyNeighbors:
            if neighbor not in frontier and neighbor not in cluster:
                frontier.append(neighbor)

    return cluster


# Find a cluster and determine the number of liberties by querying the number of adjacent tiles to all stones in the
# cluster that are not occupied. Liberties are used for heuristic calculations
def numClusterLiberties(board, point):
    numLiberties = 0
    friendlyCluster = findFriendlyCluster(board, point)
    for friendlyPoint in friendlyCluster:
        neighbors = findNeighbors(friendlyPoint)
        for neighbor in neighbors:
            if board[neighbor[0]][neighbor[1]] == 0:
                numLiberties += 1

    return numLiberties


# Determine location of dead stones by determining if the number of liberties of the cluster is 0. Used to remove the
# stones to predict future board configurations
def findDeadStones(board, color):
    dead = list()
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if board[rowInd][colInd] == color and (rowInd, colInd) not in dead:
                numLiberties = numClusterLiberties(board, (rowInd, colInd))
                if numLiberties == 0:
                    dead.append((rowInd, colInd))
    return dead


# Removes dead stones from the board
def removeDeadStones(board, color):
    deadStones = findDeadStones(board, color)
    for point in deadStones:
        board[point[0]][point[1]] = 0

    return board


# Determines if a move played is a KO by checking if the board after the move is the same as the board before the
# last move. If both boards are the same, then the move results in a KO
def isMoveKO(prevBoard, boardAfterMove):
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if prevBoard[rowInd][colInd] != boardAfterMove[rowInd][colInd]:
                return False

    return True


# Determines if placing a player stone at the point is a valid move. Check liberties and the KO rule
def isValidMove(board, prevBoard, playerColor, point):
    row, col = point[0], point[1]
    if board[row][col] != 0:
        return False
    boardCopy = copyBoard(board)
    boardCopy[row][col] = playerColor
    boardCopy = removeDeadStones(boardCopy, 3-playerColor)
    if isMoveKO(prevBoard, boardCopy):
        return False
    elif numClusterLiberties(boardCopy, point) > 0:
        return True
    else:
        return False


# Based on the board and player color, get all valid moves by looking at all unoccupied spaces and determining if
# they are valid based on liberty and KO rules
def getAllValidMoves(board, prevBoard, playerColor):
    validMoves = list()
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            point = (rowInd, colInd)
            if isValidMove(board, prevBoard, playerColor, point):
                validMoves.append(point)
    return validMoves


# Heuristic function that determines the utility of a move by calculating:
# SUM(Player Stones + ClusterLiberty*NumberOfStonesInCluster)
# For example, if there is one cluster containing 5 stones, with a liberty of 3:
# SUM(5 + 3*5) = 20
# Used for the minimax algorithm to determine utility end states
def utilityFunction(board, playerColor):
    global PLAYER_COLOR
    playerScore = 0
    opponentScore = 0
    playerUtility = 0
    opponentUtility = 0

    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if board[rowInd][colInd] == PLAYER_COLOR:
                playerScore += 1
                playerUtility += playerScore + numClusterLiberties(board, (rowInd, colInd))
            elif board[rowInd][colInd] == 3 - PLAYER_COLOR:
                opponentScore += 1
                opponentUtility += opponentScore + numClusterLiberties(board, (rowInd, colInd))

    if playerColor == PLAYER_COLOR:
        return playerUtility - opponentUtility
    else:
        return opponentUtility - playerUtility


# Main minimax algorithm that utilizes the branching minimax algorithm for recursion
# Maximizes the reward (given by the heuristic) and keeps track of all moves that have equally high utility as a list
# Returns all maximum utility moves
def minimaxMain(board, prevBoard, depth, alpha, beta, playerColor):
    bestMoves = list()
    best = 0
    boardCopy = copyBoard(board)

    validMoves = getAllValidMoves(board, prevBoard, playerColor)
    for move in validMoves:
        nextBoard = copyBoard(board)
        nextBoard[move[0]][move[1]] = playerColor
        nextBoard = removeDeadStones(nextBoard, 3-playerColor)
        playerUtility = utilityFunction(nextBoard, 3-playerColor)
        bestMove = minimaxBranch(nextBoard, boardCopy, depth, alpha, beta, playerUtility, 3-playerColor)
        currentScore = -1*bestMove

        if currentScore > best or len(bestMoves) == 0:
            best = currentScore
            alpha = best
            bestMoves = [move]
        elif currentScore == best:
            bestMoves.append(move)

    return bestMoves

# Does the recursion algorithm for minimax with alpha beta pruning
# Keeps track of whether the node is min node or max node by querying player color
# Each recursive call switches the color from player to opponent
def minimaxBranch(board, prevBoard, depth, alpha, beta, utility, nextPlayerColor):
    global PLAYER_COLOR
    if depth == 0:
        return utility
    best = utility

    boardCopy = copyBoard(board)
    validMoves = getAllValidMoves(board, prevBoard, nextPlayerColor)
    for move in validMoves:
        nextBoard = copyBoard(board)
        nextBoard[move[0]][move[1]] = nextPlayerColor
        nextBoard = removeDeadStones(nextBoard, 3-nextPlayerColor)

        playerUtility = utilityFunction(nextBoard, 3-nextPlayerColor)
        bestMove = minimaxBranch(nextBoard, boardCopy, depth-1, alpha, beta, playerUtility, 3-nextPlayerColor)

        currentScore = -1*bestMove

        if currentScore > best:
            best = currentScore

        newScore = -1*best

        if nextPlayerColor == PLAYER_COLOR:
            if newScore > beta:
                return best
            elif best > alpha:
                alpha = best
        else:
            if newScore < alpha:
                return best
            elif best > beta:
                beta = best

    return best


# Determine if a board is the empty board. Used to speed up first move selection by using top right corner first
# moves (black move)
def emptyBoard(board):
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if board[rowInd][colInd] != 0:
                return False
    return True


# Determine how many pieces are currently on the board. Used to speed up second move selection by using bottom right
# corner first move (white move)
def numPieces(board):
    count = 0
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if board[rowInd][colInd] != 0:
                count += 1
    return count


# Called ONLY when there is one piece on the board, no other time. Finds the location of the first piece to speed up
# the second move (white move)
def findPiece(board):
    for rowInd in range(BOARD_SIZE):
        for colInd in range(BOARD_SIZE):
            if board[rowInd][colInd] != 0:
                return (rowInd,colInd)

# Main script execution
#start = time.perf_counter()

# read the input file
prevBoard, board = readFile()

# If the board is empty play the 2,3 move
if emptyBoard(board):
    action = "2,3"
# If the board has one piece and it's not at 3,3, play the 3,3 move
elif numPieces(board) == 1 and findPiece(board) != (3,3):
    action = "3,3"
# Otherwise we're probably at the second turn, use minimax to determine next step
else:
    # limit depth of minimax search to 2 moves (similar to aggressive player)
    bestMoves = minimaxMain(board, prevBoard, 2, -10000, -10000, PLAYER_COLOR)
    # If no best moves, PASS
    if len(bestMoves) == 0:
        action = "PASS"
    else:
        # All moves have equal utility, choose one at random since we don't want to go deeper in our search
        move = random.choice(bestMoves)
        action = str(move[0])+","+str(move[1])
# Write to output
outFile = open(OUTPUT_FILE_NAME, "w")
outFile.write(action)
outFile.close()
#end = time.perf_counter()
#print("TIME:", end-start)