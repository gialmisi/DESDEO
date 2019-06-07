from .conftest import moproblem_params, variables, variable_params, test_file_path

import numpy as np
import pytest


class TestMOProblem():
    """Documentation for test_MOProblem

    """

    def test_init(self, moproblem):
        """Test whether the attributes are initialized correctly in the __init__ method
        of MOProblem.

        """
        # arguments
        assert(moproblem.nobj == moproblem_params["nobj"])
        assert(moproblem.nconst == moproblem_params["nconst"])
        assert(np.all(np.isclose(moproblem.ideal, moproblem_params["ideal"])))
        assert(np.all(np.isclose(moproblem.nadir, moproblem_params["nadir"])))
        assert(np.all(np.equal(moproblem.maximized, moproblem_params["maximized"])))
        assert(all([a == b for (a, b) in
                    zip(moproblem.objectives, moproblem_params["objectives"])]))
        assert(moproblem.name == moproblem_params["name"])
        assert(np.all(np.equal(moproblem.points, moproblem_params["points"])))

        # other attributes
        assert(moproblem.variables == [])

    def test_objective_names(self, moproblem_no_objectives_names):
        """Test that the objectives are named properly if no names are supplied.

        """
        obj_names = ["f%i" % (i + 1) for i in range(moproblem_no_objectives_names.nobj)]
        assert(all([a == b for (a, b) in zip(
            obj_names, moproblem_no_objectives_names.objectives)]))

    def test_maximum(self, moproblem):
        """Test that the maximum values are set correctly for each objective.  That is,
        for a minimized objective, the maximum (as in best value) should be the
        ideal, and for a maximized objective, the maximum should be the negated
        nadir point (when posed as a minimization problem).

        """
        maximums = [-nadir if m else ideal for (ideal, nadir, m) in zip(
            moproblem.nadir, moproblem.ideal, moproblem.maximized)]
        assert(np.all(np.isclose(moproblem.maximum, maximums)))

    def test_minimum(self, moproblem):
        """Test that the mimimum values are set correctly for each objective.  That is,
        for a minimized objective, the minimum (as in worst value) should be the
        nadir, and for a maximized objective, the minimum should the the negated
        ideal point (when posed as a minimization problem).

        """
        minimums = [-ideal if m else nadir for (ideal, nadir, m) in zip(
            moproblem.nadir, moproblem.ideal, moproblem.maximized)]
        assert(np.all(np.isclose(moproblem.minimum, minimums)))

    def test_objective_bounds(self, moproblem, moproblem_bare):
        """Test that proper bounds are returned.

        """
        ideals, nadirs = moproblem.objective_bounds()
        assert(np.all(np.isclose(ideals, moproblem_params["ideal"])))
        assert(np.all(np.isclose(nadirs, moproblem_params["nadir"])))

        with pytest.raises(NotImplementedError):
            _, _ = moproblem_bare.objective_bounds()

    def test_nof_objectives(self, moproblem):
        """Test that the proper number of objectives is returned. NOTE: the function
        nof_objectives was changed in a recent commit. This test is therefore most
        probablbby deprecated.

        """
        assert(moproblem.nof_objectives() == moproblem_params["nobj"])

        # nadir not set
        moproblem.nadir = []
        assert(moproblem.nof_objectives() is None)

        # nadir and ideal lengths don't match
        moproblem.nadir = [1, 2, 3]
        with pytest.raises(AssertionError):
            _ = moproblem.nof_objectives()

    def test_nof_variables(self, moproblem_with_vars):
        """Test that the proper number of variables is returned.

        """
        assert(moproblem_with_vars.nof_variables() == len(variables))

    def test_variable_bounds(self, moproblem_with_vars):
        """Test that proper bounds are returned for the variables.

        """
        assert(np.all(np.isclose(
            moproblem_with_vars.bounds(),
            [v.bounds for v in variables])))


class TestVariable():
    def test_init(self, variable):
        assert(np.all(np.isclose(variable.bounds, variable_params["bounds"])))
        assert(np.isclose(variable.starting_point, variable_params["starting_point"]))
        assert(variable.name == variable_params["name"])


class TestPreGeneratedProblem():
    points = []
    with open(test_file_path) as handle:
        for line in handle:
            points.append(list(map(float, map(str.strip, line.split(',')))))

    def test_init_file(self, pregeneratedproblem_file):
        points = TestPreGeneratedProblem.points
        assert(
            np.all([
                np.all(np.isclose(a, b)) for (a, b) in zip(
                    pregeneratedproblem_file.points, points)]))
        assert(
            np.all([
                np.all(np.isclose(a, b)) for (a, b) in zip(
                    pregeneratedproblem_file.original_points, points)]))
        # min: minimum of each column -> ideal
        assert(np.all(np.isclose(
            pregeneratedproblem_file.ideal, np.min(points, axis=0))))
        # max: maximum of each column -> nadir
        assert(np.all(np.isclose(
            pregeneratedproblem_file.nadir, np.max(points, axis=0))))

    def test_init_param(self, pregeneratedproblem_param):
        points = TestPreGeneratedProblem.points
        assert(
            np.all([
                np.all(np.isclose(a, b)) for (a, b) in zip(
                    pregeneratedproblem_param.points, points)]))
        assert(
            np.all([
                np.all(np.isclose(a, b)) for (a, b) in zip(
                    pregeneratedproblem_param.original_points, points)]))
        # min: minimum of each column -> ideal
        assert(np.all(np.isclose(
            pregeneratedproblem_param.ideal, np.min(points, axis=0))))
        # max: maximum of each column -> nadir
        assert(np.all(np.isclose(
            pregeneratedproblem_param.nadir, np.max(points, axis=0))))

    def test_evalueate(self, pregeneratedproblem_file):
        points = TestPreGeneratedProblem.points
        assert(
            np.all([
                np.all(np.isclose(a, b)) for (a, b) in zip(
                    pregeneratedproblem_file.evaluate(), points)]))
