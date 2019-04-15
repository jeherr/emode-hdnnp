#include <Python.h>
#include <numpy/arrayobject.h>
#include <dictobject.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <set>

using namespace std;

#if PY_MAJOR_VERSION >= 3
    #define PyInt_FromLong PyLong_FromLong
		#define PyInt_AS_LONG PyLong_AS_LONG
#endif

struct MyComparator
{
	const std::vector<double> & value_vector;

	MyComparator(const std::vector<double> & val_vec):
	value_vector(val_vec) {}

	bool operator()(int i1, int i2)
	{
		return value_vector[i1] < value_vector[i2];
	}
};


double dist(double x0,double y0,double z0,double x1,double y1,double z1)
{
	return sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));
}


static PyObject* Make_NLTensor(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyzs;
	PyArrayObject *zs;
	double rng;
	int nreal;
	int DoPerms;
	int DoSort;
	if (!PyArg_ParseTuple(args, "O!O!diii", &PyArray_Type, &xyzs, &PyArray_Type, &zs, &rng, &nreal, &DoPerms, &DoSort))
		return NULL;
	double *xyzs_data;
	int32_t *z_data;
	xyzs_data = (double*) ((PyArrayObject*) xyzs)->data;
	z_data = (int32_t*) ((PyArrayObject*) zs)->data;
	const int nmol = (xyzs->dimensions)[0];
	const int nat = (xyzs->dimensions)[1];

	struct ind_dist {
		int ind;
		double dist;
	};
	struct by_dist {
			bool operator()(ind_dist const &a, ind_dist const &b) {
				if (a.dist > 0.01 && b.dist > 0.01)
					return a.dist < b.dist;
				else if (a.dist > 0.01)
					return true;
				else
					return false;
			}
	};

	typedef std::vector< std::vector<ind_dist> > vov;
	typedef std::vector< vov > vovov;
	vovov NLS;

	for (int k=0; k<nmol; ++k)
	{
		double* xyz_data = xyzs_data+k*(3*nat);
		std::vector<double> XX;
		XX.assign(xyz_data,xyz_data+3*nat);
		std::vector<int> y(nat);
		std::size_t n(0);
		std::generate(y.begin(), y.end(), [&]{ return n++; });
		std::sort(y.begin(),y.end(), [&](int i1, int i2) { return XX[i1*3] < XX[i2*3]; } );
		// So y now contains sorted x indices, do the skipping Neighbor list.
		vov tmp(nreal);
		for (int i=0; i< nat; ++i)
		{
			int I = y[i];
			// We always work in order of increasing X...
			for (int j=i+1; j < nat; ++j)
			{
				int J = y[j];
				if (!(I<nreal || J<nreal))
					continue;

				if (z_data[k*(nat)+I]<=0 || z_data[k*(nat)+J]<=0)
					continue;
				if (fabs(XX[I*3] - XX[J*3]) > rng)
					break;

				double dx = (xyz_data[I*3+0]-xyz_data[J*3+0]);
				double dy = (xyz_data[I*3+1]-xyz_data[J*3+1]);
				double dz = (xyz_data[I*3+2]-xyz_data[J*3+2]);
				double dij = sqrt(dx*dx+dy*dy+dz*dz) + 0.0000000000001;
				if (dij < rng)
				{
					ind_dist Id = {I,dij};
					ind_dist Jd = {J,dij};
					if (I<J)
					{
						tmp[I].push_back(Jd);
						if (J<nreal && DoPerms==1)
							tmp[J].push_back(Id);
					}
					else
					{
						tmp[J].push_back(Id);
						if (I<nreal && DoPerms==1)
							tmp[I].push_back(Jd);
					}
				}
			}
		}
		NLS.push_back(tmp);
	}
	// Determine the maximum number of neighbors and make a tensor.
	int MaxNeigh = 0;

	for (int i = 0; i<NLS.size(); ++i)
	{
		vov& tmp = NLS[i];
		for (int j = 0; j<tmp.size(); ++j)
		{
			if (tmp[j].size() > MaxNeigh)
				MaxNeigh = tmp[j].size();
			if (DoSort)
				std::sort(tmp[j].begin(), tmp[j].end(), by_dist());
		}
	}
	npy_intp outdim2[3] = {nmol,nat,MaxNeigh};
	PyObject* NLTensor = PyArray_ZEROS(3, outdim2, NPY_INT32,0);
	int32_t* NL_data = (int32_t*) ((PyArrayObject*)NLTensor)->data;
	for (int i = 0; i<nmol; ++i)
	{
		for (int j = 0; j<nat; ++j)
		{
			for (int k=0; k<MaxNeigh; ++k)
			{
				if (k < NLS[i][j].size())
					NL_data[i*(nat*MaxNeigh)+j*MaxNeigh+k] = (int32_t)(NLS[i][j][k].ind);
				else
					NL_data[i*(nat*MaxNeigh)+j*MaxNeigh+k] = (int32_t)(-1);
			}
		}
	}
	return NLTensor;
}


//
// Makes a triples tensor.
// if DoPerms = True
// Output is nMol X MaxNAtom X (MaxNeigh * MaxNeigh-1) X 2
// else:
// Output is nMol X MaxNAtom X (MaxNeigh * ((MaxN+1)/2-1)) X 2
static PyObject* Make_TLTensor(PyObject *self, PyObject  *args)
{
	PyArrayObject *xyzs;
	PyArrayObject *zs;
	double rng;
	int nreal;
	int DoPerms;
	if (!PyArg_ParseTuple(args, "O!O!dii", &PyArray_Type, &xyzs, &PyArray_Type, &zs, &rng, &nreal, &DoPerms))
		return NULL;
	double *xyzs_data;
	int32_t *z_data;
	xyzs_data = (double*) ((PyArrayObject*) xyzs)->data;
	z_data = (int32_t*) ((PyArrayObject*) zs)->data;
	const int nmol = (xyzs->dimensions)[0];
	const int nat = (xyzs->dimensions)[1];

	struct ind_dist {
		int ind;
		double dist;
	};
	struct by_dist {
			bool operator()(ind_dist const &a, ind_dist const &b) {
				if (a.dist > 0.01 && b.dist > 0.01)
					return a.dist < b.dist;
				else if (a.dist > 0.01)
					return true;
				else
					return false;
			}
	};

	typedef std::vector< std::vector<ind_dist> > vov;
	typedef std::vector< vov > vovov;
	vovov NLS;

	for (int k=0; k<nmol; ++k)
	{
		double* xyz_data = xyzs_data+k*(3*nat);
		std::vector<double> XX;
		XX.assign(xyz_data,xyz_data+3*nat);
		std::vector<int> y(nat);
		std::size_t n(0);
		std::generate(y.begin(), y.end(), [&]{ return n++; });
		std::sort(y.begin(),y.end(), [&](int i1, int i2) { return XX[i1*3] < XX[i2*3]; } );
		// So y now contains sorted x indices, do the skipping Neighbor list.
		vov tmp(nreal);
		for (int i=0; i< nat; ++i)
		{
			int I = y[i];
			// We always work in order of increasing X...
			for (int j=i+1; j < nat; ++j)
			{
				int J = y[j];
				if (!(I<nreal || J<nreal))
					continue;

				if (z_data[k*(nat)+I]<=0 || z_data[k*(nat)+J]<=0)
					continue;
				if (fabs(XX[I*3] - XX[J*3]) > rng)
					break;

				double dx = (xyz_data[I*3+0]-xyz_data[J*3+0]);
				double dy = (xyz_data[I*3+1]-xyz_data[J*3+1]);
				double dz = (xyz_data[I*3+2]-xyz_data[J*3+2]);
				double dij = sqrt(dx*dx+dy*dy+dz*dz) + 0.0000000000001;
				if (dij < rng)
				{
					ind_dist Id = {I,dij};
					ind_dist Jd = {J,dij};
					if (I<J)
					{
						tmp[I].push_back(Jd);
						if (J<nreal)
							tmp[J].push_back(Id);
					}
					else
					{
						tmp[J].push_back(Id);
						if (I<nreal)
							tmp[I].push_back(Jd);
					}
				}
			}
		}
		NLS.push_back(tmp);
	}
	// Determine the maximum number of neighbors and make a tensor.
	int MaxNeigh = 0;
	for (int i = 0; i<NLS.size(); ++i)
	{
		vov& tmp = NLS[i];
		for (int j = 0; j<tmp.size(); ++j)
		{
			if (tmp[j].size() > MaxNeigh)
				MaxNeigh = tmp[j].size();
			std::sort(tmp[j].begin(), tmp[j].end(), by_dist());
		}
	}

	int Dim2 = MaxNeigh*(MaxNeigh-1);
	if (!DoPerms)
		Dim2 = MaxNeigh*(MaxNeigh+1)/2 - MaxNeigh;

	npy_intp outdim2[4] = {nmol,nat,Dim2,2};
	PyObject* NLTensor = PyArray_ZEROS(4, outdim2, NPY_INT32,0);
	int32_t* NL_data = (int32_t*) ((PyArrayObject*)NLTensor)->data;
	memset(NL_data, -1, sizeof(int32_t)*nmol*nat*Dim2*2);
	for (int i = 0; i<nmol; ++i)
	{
		for (int j = 0; j<nat; ++j)
		{
			int counter = 0;
			for (int k=0; k<MaxNeigh; ++k)
			{
				if (k < NLS[i][j].size())
				{
					for (int l=0; l<MaxNeigh; ++l)
					{
						if (l < NLS[i][j].size())
						{
							if (DoPerms && k!=l)
							{
								NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2] = (int32_t)(NLS[i][j][k].ind);
								NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2+1] = (int32_t)(NLS[i][j][l].ind);
								counter++;
							}
							else if (k<l)
							{
								NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2] = (int32_t)(NLS[i][j][k].ind);
								NL_data[i*(nat*Dim2*2)+j*(Dim2*2)+counter*2+1] = (int32_t)(NLS[i][j][l].ind);
								counter++;
							}
						}
					}
				}
			}
		}
	}
	return NLTensor;
}

struct module_state {
	PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
	#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
	#define GETSTATE(m) (&_state)
	static struct module_state _state;
#endif

static PyObject * error_out(PyObject *m) {
		struct module_state *st = GETSTATE(m);
		PyErr_SetString(st->error, "something bad happened");
		return NULL;
}

static PyMethodDef EmbMethods[] =
{
	{"Make_NLTensor", Make_NLTensor, METH_VARARGS,
	"Make_NLTensor method"},
	{"Make_TLTensor", Make_TLTensor, METH_VARARGS,
	"Make_TLTensor method"},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

	static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
	    Py_VISIT(GETSTATE(m)->error);
	    return 0;
	}

	static int myextension_clear(PyObject *m) {
	    Py_CLEAR(GETSTATE(m)->error);
	    return 0;
	}

  static struct PyModuleDef moduledef = {
		PyModuleDef_HEAD_INIT,
		"NL",     /* m_name */
		"A CAPI for TensorMol",  /* m_doc */
		sizeof(struct module_state),
		EmbMethods,    /* m_methods */
		NULL,                /* m_reload */
		myextension_traverse,                /* m_traverse */
		myextension_clear,                /* m_clear */
		NULL                /* m_free */
		};
	#pragma message("Compiling NL for Python3x")
	#define INITERROR return NULL
	PyMODINIT_FUNC
	PyInit_NL(void)
	{
		PyObject *m = PyModule_Create(&moduledef);
		if (m == NULL)
			INITERROR;
		struct module_state *st = GETSTATE(m);
		st->error = PyErr_NewException("NL.Error", NULL, NULL);
		if (st->error == NULL) {
			Py_DECREF(m);
			INITERROR;
		}
		import_array();
		return m;
	}
#else
	PyMODINIT_FUNC
	initNL(void)
	{
		(void) Py_InitModule("NL", EmbMethods);
		/* IMPORTANT: this must be called */
		import_array();
		return;
	}
#endif
