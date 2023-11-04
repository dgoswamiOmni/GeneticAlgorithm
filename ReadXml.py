import xml.etree.ElementTree as ET
import numpy as np

def read_xml(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Determine the number of vertices
    number_of_vertices = len(root.find('.//graph').findall('vertex'))

    # Initialize the distance matrix with zeros
    D = np.zeros((number_of_vertices, number_of_vertices))

    source_vertex = 0
    data = []

    # Iterate over vertex elements and populate the distance matrix
    for vertex in root.find('.//graph').findall('vertex'):
        edges = [(int(edge.text), float(edge.get('cost'))) for edge in vertex.findall('edge')]

        for dest_vertex, cost in edges:
            # print(source_vertex,dest_vertex,cost)
            D[source_vertex, dest_vertex] = cost
            D[dest_vertex, source_vertex] = cost  # Assuming the distance matrix is symmetric

        source_vertex += 1

    return D
