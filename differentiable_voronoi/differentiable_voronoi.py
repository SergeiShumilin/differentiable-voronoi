import torch
import numpy as np
from trimesh import Trimesh
from scipy.spatial import Delaunay
import matplotlib.patches
from shapely import Point
import copy
from shapely import Polygon
import matplotlib.pyplot as plt
from copy import deepcopy


class PolygonClipper:

    def __init__(self, warn_if_empty=True):
        self.warn_if_empty = warn_if_empty

    def is_inside(self, p1, p2, q):
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
        if R <= 0:
            return True
        else:
            return False

    def compute_intersection(self, p1, p2, p3, p4):

        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.

        To compute the point of intersection of the two lines, equate
        the two line equations together

        m1 * x + b1 = m2 * x + b2

        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.

        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """
        eps = 1e-4
        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0] + eps)
            b2 = p3[1] - m2 * p3[0]

            # y-coordinate of intersection
            y = m2 * x + b2

        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]

            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0] + eps)
            b1 = p1[1] - m1 * p1[0]

            # y-coordinate of intersection
            y = m1 * x + b1

        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0] + eps)
            b1 = p1[1] - m1 * p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0] + eps)
            b2 = p3[1] - m2 * p3[0]

            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2 + eps)

            # y-coordinate of intersection
            y = m1 * x + b1

        # need to unsqueeze so torch.cat doesn't complain outside func
        intersection = torch.stack((x, y)).unsqueeze(0)

        return intersection

    def clip(self, subject_polygon, clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon
        # subject_polygon and clipping_polygon are N x 2 and M x 2 torch
        # tensors respectively

        final_polygon = torch.clone(subject_polygon)

        for i in range(len(clipping_polygon)):

            # stores the vertices of the next iteration of the clipping procedure
            # final_polygon consists of list of 1 x 2 tensors
            next_polygon = torch.clone(final_polygon)

            # stores the vertices of the final clipped polygon. This will be
            # a K x 2 tensor, so need to initialize shape to match this
            final_polygon = torch.empty((0, 2))

            # these two vertices define a line segment (edge) in the clipping
            # polygon. It is assumed that indices wrap around, such that if
            # i = 0, then i - 1 = M.
            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]

            for j in range(len(next_polygon)):

                # these two vertices define a line segment (edge) in the subject
                # polygon
                s_edge_start = next_polygon[j - 1]
                s_edge_end = next_polygon[j]

                if self.is_inside(c_edge_start, c_edge_end, s_edge_end):
                    if not self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                        intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                        final_polygon = torch.cat((final_polygon, intersection), dim=0)
                    final_polygon = torch.cat((final_polygon, s_edge_end.unsqueeze(0)), dim=0)
                elif self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                    intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                    final_polygon = torch.cat((final_polygon, intersection), dim=0)

        return final_polygon


def center_of_circumcircle(pos, simplices):
    a1, b1, c1 = pos[:, 0][simplices].T
    a2, b2, c2 = pos[:, 1][simplices].T
    a1sq = a1 * a1
    a2sq = a2 * a2
    b1sq = b1 * b1
    b2sq = b2 * b2
    c1sq = c1 * c1
    c2sq = c2 * c2
    D = 2 * ((a1 - c1) * (b2 - c2) - (b1 - c1) * (a2 - c2))
    v1 = (a1sq - c1sq + a2sq - c2sq) * (b2 - c2) - (b1sq - c1sq + b2sq - c2sq) * (a2 - c2)
    v1 /= D
    v2 = (b1sq - c1sq + b2sq - c2sq) * (a1 - c1) - (a1sq - c1sq + a2sq - c2sq) * (b1 - c1)
    v2 /= D
    return torch.cat([v1.view(-1, 1), v2.view(-1, 1)], dim=1)


def triangulate(coords):
    tri = Delaunay(coords)
    mesh = Trimesh(coords, tri.simplices)

    edges_unique_faces = {}

    for i, (a, b, c) in enumerate(mesh.faces_unique_edges):
        for k in [a, b, c]:
            if not k in edges_unique_faces:
                edges_unique_faces[k] = [i]
            else:
                edges_unique_faces[k].append(i)

    mesh.edges_unique_faces = edges_unique_faces

    boundary_edges = np.array([k for k, v in edges_unique_faces.items() if len(v) == 1])
    boundary_nodes = torch.tensor(np.unique(np.array(mesh.edges_unique[boundary_edges]).flatten()))
    boundary_faces = [v[0] for k, v in edges_unique_faces.items() if len(v) == 1]

    border_faces_border_edges = {}
    for k, v in edges_unique_faces.items():
        for face in v:
            if face in boundary_faces and k in boundary_edges:
                if not face in border_faces_border_edges.keys():
                    border_faces_border_edges[face] = [k, ]
                else:
                    border_faces_border_edges[face].append(k)

    mesh.border_faces_border_edges = border_faces_border_edges

    border_faces_border_nodes = deepcopy(border_faces_border_edges)
    for k, v in border_faces_border_edges.items():
        border_faces_border_nodes[k] = mesh.edges_unique[border_faces_border_edges[k]]

    mesh.border_faces_border_nodes = border_faces_border_nodes

    vertex_faces = torch.tensor(mesh.vertex_faces)
    vertex_faces = torch.where(vertex_faces == -1, vertex_faces[:, 0].reshape(-1, 1).expand(vertex_faces.shape),
                               vertex_faces)

    mesh.vertex_faces_ext = vertex_faces
    mesh.boundary_edges = boundary_edges
    mesh.boundary_nodes = boundary_nodes
    mesh.boundary_faces = boundary_faces

    return mesh


def orient_vertex_faces(coords, vertex_faces, circumcenters, boundary_nodes):
    def orient_points(points):
        cx, cy = np.unique(points, axis=0).mean(0)
        x, y = points.T
        angles = np.arctan2(x - cx, y - cy)
        # indices = np.argsort(-angles)
        return angles

    circumcenters, vertex_faces = complete_boundary_cells(coords, vertex_faces, circumcenters, boundary_nodes)
    wjdj = circumcenters[vertex_faces]
    angles = np.array([orient_points(line.detach().numpy()) for line in wjdj])
    assert angles.shape == vertex_faces.shape
    return circumcenters, torch.tensor([[x for x, _ in sorted(zip(idx, angle), key=lambda pair: pair[1])]
                                        for angle, idx in zip(angles, vertex_faces)])


def complete_boundary_cells(coords, vertex_faces_dict, circumcenters, boundary_nodes, mesh):
    circumcenters_orig_len = len(circumcenters)
    vertex_faces_dict_copy = copy.deepcopy(vertex_faces_dict)
    center = coords[~boundary_nodes].mean(dim=0)

    for boundary_node in boundary_nodes:  # if vertex in convex hull
        faces = vertex_faces_dict_copy[int(boundary_node.item())]
        for face in faces:
            if face in mesh.boundary_faces:
                neigh_vertices = mesh.border_faces_border_nodes[face]
                for edge in range(neigh_vertices.shape[0]):
                    v1, v2 = neigh_vertices[edge, 0], neigh_vertices[edge, 1]
                    if v1 == boundary_node or v2 == boundary_node:  # avoid "moving" to neighboring voronoi cells

                        p1 = coords[v1]
                        p2 = coords[v2]

                        edge_vector = p2 - p1
                        orig_n = edge_vector / torch.norm(edge_vector)
                        n = torch.tensor([-orig_n[1], orig_n[0]], dtype=coords.dtype)

                        # Calculate the midpoint of the edge
                        midpoint = 0.5 * (p1 + p2)
                        direction = torch.sign((midpoint - center).dot(n)) * n
                        # Create the far away point by moving from the midpoint in the direction of n
                        far_away_point = midpoint + 10.0 * direction  # The large number simulates "infinity"

                        circumcenters = torch.cat([circumcenters, far_away_point[None, :]], dim=0)
                        vertex_faces_dict[int(boundary_node.item())].append(circumcenters.shape[0] - 1)

    return circumcenters, vertex_faces_dict


def who_is_inside_the_boundary(circumcenters, boundary):
    with torch.no_grad():
        B = Polygon(boundary)
        inside = []
        for i, c in enumerate(circumcenters):
            if B.contains(Point(c)):
                inside.append(i)
        return inside


def convert_vertex_faces_dict_into_circumcenters_vertex_dict(vertex_faces_dict):
    circumcenters_vertex_dict = {}
    for v, cc in vertex_faces_dict.items():
        for c in cc:
            if c not in circumcenters_vertex_dict.keys():
                circumcenters_vertex_dict[c] = {v}
            else:
                circumcenters_vertex_dict[c].add(v)
    return circumcenters_vertex_dict


def select_only_outside_vertices(indicies_of_outside_circums, circumcenters_vertex_dict):
    for i in indicies_of_outside_circums:
        del circumcenters_vertex_dict[i]
    return circumcenters_vertex_dict


def retrieve_the_vertices(dict_of_circums_outside_the_border):
    vertices_to_crop = set()
    for k, v in dict_of_circums_outside_the_border.items():
        for _ in v:
            vertices_to_crop.add(_)
    return set(vertices_to_crop)


def select_edges_adjacent_to_the_given_vetrices(vertices, edges_unique):
    edges = []
    for k, (i, j) in enumerate(edges_unique):
        if i in vertices and j in vertices:
            edges.append(k)
    return set(edges)


def orient_vertex_faces_dict(coords, vertex_faces_dict, circumcenters, boundary_nodes):
    def orient_points(points):
        cx, cy = np.unique(points, axis=0).mean(0)
        x, y = points.T
        angles = np.arctan2(x - cx, y - cy)
        return -angles

    vertex_faces_dict = {
        vertex: [x for _, x in sorted(zip(orient_points(circumcenters[vertex_faces_dict[vertex]].detach().numpy()),
                                          vertex_faces_dict[vertex]), key=lambda pair: pair[0])]
        for vertex in vertex_faces_dict.keys()}
    return circumcenters, vertex_faces_dict


def calc_areas(circumcenters, vertex_faces):
    x_cell_borders = circumcenters[:, 0][vertex_faces]
    y_cell_borders = circumcenters[:, 1][vertex_faces]
    S1 = torch.sum(x_cell_borders * torch.roll(y_cell_borders, -1, 1), dim=1)
    S2 = torch.sum(y_cell_borders * torch.roll(x_cell_borders, -1, 1), dim=1)
    return .5 * torch.abs(S1 - S2)


def calc_areas_dict(circumcenters, vertex_faces_dict):
    areas_tensor = torch.zeros((len(vertex_faces_dict.keys())))
    for vertex in vertex_faces_dict.keys():
        x_cell_borders = circumcenters[:, 0][vertex_faces_dict[vertex]]
        y_cell_borders = circumcenters[:, 1][vertex_faces_dict[vertex]]
        S1 = torch.sum(x_cell_borders * torch.roll(y_cell_borders, -1, 0))
        S2 = torch.sum(y_cell_borders * torch.roll(x_cell_borders, -1, 0))
        areas_tensor[vertex] = .5 * torch.abs(S1 - S2)
    return areas_tensor


def calc_areas_dict_bounded(
        clipped_vertices_dict):
    areas_tensor = torch.zeros((len(clipped_vertices_dict.keys())))
    for vertex in clipped_vertices_dict.keys():
        x_cell_borders = clipped_vertices_dict[vertex][:, 0]
        y_cell_borders = clipped_vertices_dict[vertex][:, 1]
        S1 = torch.sum(x_cell_borders * torch.roll(y_cell_borders, -1, 0))
        S2 = torch.sum(y_cell_borders * torch.roll(x_cell_borders, -1, 0))
        areas_tensor[vertex] = .5 * torch.abs(S1 - S2)
    return areas_tensor


def is_points_close(p1, p2, eps=1e-2):
    return (p1 - p2).norm() < eps


def find_the_common_side_of_the_two_voronoi_regions(region1, region2):
    assert len(region1) > 2
    assert len(region2) > 2

    for i in range(-1, len(region1) - 1):
        p1, p2 = region1[i], region1[i + 1]
        for j in range(-1, len(region2) - 1):
            p3, p4 = region2[j], region2[j + 1]
            if (is_points_close(p1, p3) and is_points_close(p2, p4)) or (
                    is_points_close(p1, p4) and is_points_close(p2, p3)):
                return True, (p1 - p2).norm()

    return False, -1


def calc_lengths_of_voronoi_edges_adjacent_to_vertices_to_crop(edges,
                                                               clipped_vertices_dict,
                                                               edges_unique):
    lengths_of_voronoi_edges = torch.zeros(len(edges_unique))
    edges_to_delete = np.full(len(edges_unique), False)

    for e in edges:
        i, j = edges_unique[e]
        region1 = clipped_vertices_dict[i]
        region2 = clipped_vertices_dict[j]
        is_adjacent, length = find_the_common_side_of_the_two_voronoi_regions(region1, region2)
        if is_adjacent:
            lengths_of_voronoi_edges[e] = length
        else:
            edges_to_delete[e] = True
    return lengths_of_voronoi_edges, edges_to_delete


def calc_lengths_of_inner_voronoi_edges(inner_edges, circumcenters, lengths_of_voronoi_edges, edges_unique_faces):
    """Calculate the lengths of those Voronoi edges that are not croped by the boundary."""

    for e in inner_edges:
        c = edges_unique_faces[e]
        if len(c) == 2:
            c1, c2 = c
            length = (circumcenters[c1] - circumcenters[c2]).norm()
            lengths_of_voronoi_edges[e] = length
        elif len(c) == 1:
            c1 = c
            length = circumcenters[c1].norm()
            lengths_of_voronoi_edges[e] = 10 * length
    return lengths_of_voronoi_edges


def crop_cells(coords, vertex_faces_dict, circumcenters, boundary):
    circums_inside_the_boundary = who_is_inside_the_boundary(circumcenters, boundary)
    circumcenters_vertex_dict = convert_vertex_faces_dict_into_circumcenters_vertex_dict(vertex_faces_dict)
    only_outside_vertices = select_only_outside_vertices(circums_inside_the_boundary,
                                                         circumcenters_vertex_dict)
    vertices_to_crop = retrieve_the_vertices(only_outside_vertices)
    keys = vertex_faces_dict.keys()
    vertices_to_keep = set(keys) - vertices_to_crop
    assert len(keys) == len(coords)
    assert len(keys) == (len(vertices_to_keep) + len(vertices_to_crop))
    assert len(vertices_to_keep.intersection(vertices_to_crop)) == 0
    assert len(coords) == (len(vertices_to_keep) + len(vertices_to_crop))

    clipped_vertices_dict = {}
    for vertex in vertices_to_crop:
        cell = circumcenters[vertex_faces_dict[vertex]]
        clipper = PolygonClipper(warn_if_empty=False)
        clipped_polygon = clipper.clip(cell, boundary)
        cx, cy = torch.unique(clipped_polygon, dim=0).mean(0)
        x, y = clipped_polygon.T
        angles = torch.arctan2(x - cx, y - cy)
        indices = torch.argsort(-angles)
        clipped_vertices_dict[vertex] = clipped_polygon[indices]

    for vertex in vertices_to_keep:
        clipped_vertices_dict[vertex] = circumcenters[vertex_faces_dict[vertex]]

    return clipped_vertices_dict, vertices_to_keep, vertices_to_crop


def draw_voronoi_region(coords, line, vertex, ax, color='g'):
    p = matplotlib.patches.Polygon(line, facecolor=color, edgecolor='black', alpha=0.3)
    ax.scatter(coords[vertex][0], coords[vertex][1], c='black')
    ax.text(coords[vertex][0], coords[vertex][1], str(vertex), c=color)
    ax.add_patch(p)


def differentiable_voronoi(coords, mesh, boundary=None, vizualize=False):
    vertex_faces_dict = {}
    for i, row in enumerate(mesh.vertex_faces):
        vertex_faces_dict[i] = [face for face in row if face != -1]

    assert len(vertex_faces_dict) == len(coords), print(len(vertex_faces_dict), len(coords))

    circumcenters = center_of_circumcircle(coords, torch.tensor(mesh.faces))
    clipped_vertices_dict = None

    circumcenters, vertex_faces_dict = complete_boundary_cells(coords, vertex_faces_dict, circumcenters,
                                                               mesh.boundary_nodes, mesh)
    assert len(vertex_faces_dict) == len(coords), print(len(vertex_faces_dict), len(coords))
    circumcenters, vertex_faces_dict = orient_vertex_faces_dict(coords, vertex_faces_dict, circumcenters,
                                                                mesh.boundary_nodes)
    assert len(vertex_faces_dict) == len(coords)

    if boundary is not None:
        clipped_vertices_dict, voronoi_region_to_keep, voronoi_regions_to_crop = crop_cells(coords, vertex_faces_dict,
                                                                                            circumcenters, boundary)
        assert len(clipped_vertices_dict) == len(coords)
        edges_adjacent_to_vertices_to_crop = select_edges_adjacent_to_the_given_vetrices(voronoi_regions_to_crop,
                                                                                         mesh.edges_unique)
        lengths_of_voronoi_edges, edges_to_delete = calc_lengths_of_voronoi_edges_adjacent_to_vertices_to_crop(
            edges_adjacent_to_vertices_to_crop,
            clipped_vertices_dict,
            mesh.edges_unique)

        lengths_of_voronoi_edges = calc_lengths_of_inner_voronoi_edges(
            set(list(range(len(mesh.edges_unique)))) - edges_adjacent_to_vertices_to_crop,
            circumcenters,
            lengths_of_voronoi_edges,
            mesh.edges_unique_faces)
        lengths_of_voronoi_edges = lengths_of_voronoi_edges[~edges_to_delete]
        areas = calc_areas_dict_bounded(clipped_vertices_dict)
    else:
        edges_to_delete = np.full(len(mesh.edges_unique), False)
        areas = calc_areas_dict(circumcenters, vertex_faces_dict)
        lengths_of_voronoi_edges = torch.zeros((len(mesh.edges_unique)))
        lengths_of_voronoi_edges = calc_lengths_of_inner_voronoi_edges(set(list(range(len(mesh.edges_unique)))),
                                                                       circumcenters,
                                                                       lengths_of_voronoi_edges,
                                                                       mesh.edges_unique_faces)

    assert len(mesh.edges_unique[~edges_to_delete]) == len(lengths_of_voronoi_edges)
    assert len(areas[areas == 0]) == 0

    if vizualize:
        with torch.no_grad():
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot()
            if boundary is not None:
                for vertex in voronoi_region_to_keep:
                    line = clipped_vertices_dict[vertex]
                    draw_voronoi_region(coords, line, vertex, ax)
                for vertex in voronoi_regions_to_crop:
                    line = clipped_vertices_dict[vertex]
                    draw_voronoi_region(coords, line, vertex, ax, color='r')
            else:
                for vertex in vertex_faces_dict.keys():
                    line = circumcenters[vertex_faces_dict[vertex]].clone().detach()
                    draw_voronoi_region(coords, line, vertex, ax)
            plt.show()

    return torch.tensor(mesh.edges_unique[~edges_to_delete].T), areas, lengths_of_voronoi_edges, clipped_vertices_dict
