/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_jpy_PyLib_Diag */

#ifndef _Included_org_jpy_PyLib_Diag
#define _Included_org_jpy_PyLib_Diag
#ifdef __cplusplus
extern "C" {
#endif
#undef org_jpy_PyLib_Diag_F_OFF
#define org_jpy_PyLib_Diag_F_OFF 0L
#undef org_jpy_PyLib_Diag_F_TYPE
#define org_jpy_PyLib_Diag_F_TYPE 1L
#undef org_jpy_PyLib_Diag_F_METH
#define org_jpy_PyLib_Diag_F_METH 2L
#undef org_jpy_PyLib_Diag_F_EXEC
#define org_jpy_PyLib_Diag_F_EXEC 4L
#undef org_jpy_PyLib_Diag_F_MEM
#define org_jpy_PyLib_Diag_F_MEM 8L
#undef org_jpy_PyLib_Diag_F_JVM
#define org_jpy_PyLib_Diag_F_JVM 16L
#undef org_jpy_PyLib_Diag_F_ERR
#define org_jpy_PyLib_Diag_F_ERR 32L
#undef org_jpy_PyLib_Diag_F_ALL
#define org_jpy_PyLib_Diag_F_ALL 255L
/*
 * Class:     org_jpy_PyLib_Diag
 * Method:    getFlags
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_jpy_PyLib_00024Diag_getFlags
  (JNIEnv *, jclass);

/*
 * Class:     org_jpy_PyLib_Diag
 * Method:    setFlags
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_org_jpy_PyLib_00024Diag_setFlags
  (JNIEnv *, jclass, jint);

#ifdef __cplusplus
}
#endif
#endif
