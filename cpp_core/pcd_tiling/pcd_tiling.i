 /* Bonjour.i */
 %module pcd_tiling

 %{
	#define SWIG_FILE_WITH_INIT
 	#include "pcd_tiling.h"
 %}

 %include std_string.i
 %include std_vector.i

 // %include "numpy.i"

 %include "pcd_tiling.h"


