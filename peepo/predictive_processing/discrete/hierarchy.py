class Hierarchy:
    def __init__(self, graph, regions, inputs):
        """
        Predictive Processing Hierarchy:

        :param graph: Graph structure of the Hiearchy. Should contain
         one reserved name 'root' to indicate the root nodes. E.g.:
         graph = {
            'root': ['A'],
            'A': ['B', 'C'],
            'B': [],
            'C': []
         }

        :param regions: Dict of Region objects, identified by name.

        :param inputs: Values of the actual inputs.
         The PP Hierarchy will try to resolve any mismatch that
         exists between its predictions and the actual inputs.
         Can be likened to input from the lowest regions in our body,
         e.g. retina input.

        :type graph: dict
        :type regions: dict
        :type inputs: dict
        """
        self.graph = graph
        self.regions = regions
        self.inputs = inputs
        self.rootnames = graph.get('root')
        self.maxiter = 10

    def start(self):
        """
        Initiates the flow through the PP Hierarchy
        """
        for rootname in self.rootnames:
            root = self.regions.get(rootname)
            self.flow(root, root.getHyp(), [])

    def flow(self, node, hyp, anc):
        """
        Flows through the whole PP Hierarchy. Passing the predictions
        of the current node to its child node as its hypotheses.
        If a node has no children left, then the resolve flow is initiated,
        in case there are any prediction errors to resolve.

        :param node: Region node for which to calculate predictions.

        :param hyp: Hypotheses for current Region node. Given by either
         the parent node predictions or as manual input (for the root nodes).

        :param anc: List of ancestor nodes that should be tracked in order to
         resolve the errors in a bottom node.

        :type node: Region
        :type hyp: numpy.array
        :type anc: list
        """
        node.setHyp(hyp)
        name = node.__getattribute__('name')
        pred = node.predict()
        children = self.graph.get(name)

        if children:
            anc.append(node)
            for child in children:
                self.flow(self.regions.get(child), pred, list(anc))
        else:
            self.resolve(node, pred, self.inputs.get(name), anc)

    def resolve(self, node, pred, input, anc):
        """
        Works through the list of passed ancestors in case a bottom node
        has a prediction error, resolving the errors one by one, moving
        upwards through the ancestor list.

        :param node: Region node for which to correct errors if any.

        :param pred: Array of predicted values.

        :param input: Array of actual values.

        :param anc: List of ancestor nodes to work upward through in case
         of a bottom node error.

        :type node: Region
        :type pred: numpy.array
        :type input: numpy.array
        :type anc: list
        """
        if node.error(pred, input):
            for x in range(0, self.maxiter):
                node.update(input)
                if not node.error(node.predict(), input):
                    break
            if anc:
                parent = anc.pop()
                self.resolve(parent, parent.predict(), node.getHyp(), anc)
