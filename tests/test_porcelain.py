import pytest
import numpy as np


class TestObjective():
    def test_init(self, min_objective, max_objective, def_objective):
        """Test the initializaion of Objective

        """
        assert(min_objective.name == "min_objective")
        assert(not min_objective.maximized)
        assert(min_objective.ideal is None)
        assert(min_objective.nadir is None)

        assert(max_objective.maximized)

        assert(np.all(np.isclose(
            def_objective.ideal, [1, 2, 3])))

        assert(np.all(np.isclose(
            def_objective.nadir, [5, 6, 7])))

    def test_call(self, min_objective, max_objective, def_objective):
        """Test that the objectives are called properly.

        """
        assert(np.isclose(min_objective.inner(5, 5), 0))
        assert(np.isclose(max_objective.inner(10, 20, 30, 40), 100))
        assert(np.isclose(def_objective.inner(3, 6, 3), 6))


class TestVariable():
    pass
