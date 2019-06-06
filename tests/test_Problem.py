from .conftest import moproblem_params

import numpy as np


class TestMOProblem():
    """Documentation for test_MOProblem

    """

    def test_init(self, moproblem):
        """Test whether the attributes are initialized correctly in the __init__
        method of MOProblem."""
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

        # TODO: What does line 65 in Problem.py do? Test it!
