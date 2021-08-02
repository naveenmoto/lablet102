import numpy as np
from functools import reduce

def euclidean(node1, node2):
    x1,y1 = node1
    x2,y2 = node2
    dist = np.sqrt(pow((y2-y1),2) + pow((x2-x1),2))
    return dist

def make_route(coords):
	#find the corner point and plot it
	corners = []
	x,y = zip(*coords)
	for x1,x2,x3, y1,y2,y3 in zip(x[:-2],x[1:-1],x[2:],y[:-2],y[1:-1],y[2:]):
		slope = (x2-x1)*(y3-y2) - (x3-x2)*(y2-y1)
		if np.abs(slope) > 0.0:
			if slope > 0:
				corners.append([x2, y2,90])
			else:
				corners.append([x2, y2,-90])
			
	# Fix a turn radius r
	adj_coords = corners[1:]
	org_coords = corners[:-1]
	dists = [euclidean(n1[:-1],n2[:-1]) for n1,n2 in zip(org_coords,adj_coords)]
	if dists:
		smallest_dist = reduce(lambda a,b : a if a < b else b,dists)
		r = smallest_dist/2
	else:
		r = 1
		
	first_flag = 1
	# Shorten the straight segments by r
	# convert this into {("straight", s1), ("turn", +/- 90), ("straight", s2)}
	route = []
	for i in range(len(corners)):
		if first_flag == 1:
			first_flag = 0
			dist = euclidean(corners[i][:-1],(x[0],y[0])) - r
		else:
			dist = dists[i-1] - 2*r
		if dist > 0:
			route.append(['straight',dist])
		route.append(['turn',corners[i][-1]])
	dist = euclidean(corners[-1][:-1],(x[-1],y[-1])) - r
	route.append(['straight',dist])
	return route,corners,r