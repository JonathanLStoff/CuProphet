#!/bin/csh
setenv TBBROOT "D:\a\prophet\prophet\python\build\lib.win-amd64-cpython-39\prophet\stan_model\cmdstan-2.26.1\stan\lib\stan_math\lib\tbb_2019_U8"
setenv TBB_ARCH_PLATFORM "intel64\mingw8.3.0"
setenv TBB_TARGET_ARCH "intel64"
setenv CPATH "${TBBROOT}\include;$CPATH"
setenv LIBRARY_PATH "D:\a\prophet\prophet\python\build\lib.win-amd64-cpython-39\prophet\stan_model\cmdstan-2.26.1\stan\lib\stan_math\lib\tbb;$LIBRARY_PATH"
setenv PATH "D:\a\prophet\prophet\python\build\lib.win-amd64-cpython-39\prophet\stan_model\cmdstan-2.26.1\stan\lib\stan_math\lib\tbb;$PATH"
setenv LD_LIBRARY_PATH "D:\a\prophet\prophet\python\build\lib.win-amd64-cpython-39\prophet\stan_model\cmdstan-2.26.1\stan\lib\stan_math\lib\tbb;$LD_LIBRARY_PATH"
