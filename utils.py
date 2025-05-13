def get_edge_attribute_dict(data, attr_name):
    return {
        edge_type: data[edge_type][attr_name]
        for edge_type in data.edge_types
        if attr_name in data[edge_type]
    }

def get_node_attribute_dict(data, attr_name):
    return {
        node_type: data[node_type][attr_name]
        for node_type in data.node_types
        if attr_name in data[node_type]
    }