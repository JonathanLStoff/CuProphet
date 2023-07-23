<center><span style="font-size:25px"> CuProphet ||</span> <span style="font-size:18px">Prophet built to use CUDA</span>

![Build](https://github.com/facebook/prophet/workflows/Build/badge.svg)

_____
</center>
<center><img src="file.jpg" alt="prophet" width="400"></center>



# Compile instructions:

* install python 3.9.8
* Run these:

    
        cd [insert Dir]/CuProphet-Master
        python -m cmdstanpy.install_cmdstan(version="2.32.1", dir=[put dir here], verbose=True, cores=-1)
* Add to PATH:
    
        ~.../Users/[user]/.cmdstan/RTools40/mingw64/bin
        ~.../Users/[user]/.cmdstan/RTools40/usr/bin
* Remove any other compilers from user path and system path
* Merge these folders from the *replace* folder: \

        ~.../Users/{user}/.cmdstan/\
        ~.../Python39/Lib/site-packages/prophet/\
        ~.../Python39/Lib/site-packages/cmdstanpy/\
* Run these commands:

        cd ".\CuProphet-Master"
        python setup.py install
        cd ".\CuProphet-Master\hppfiles"
        nvcc -o cuda_func_help.dll --shared -v -arch=sm_86 --fmad=false -g cuda_func_help.cu


[![Donate with PayPal](https://raw.githubusercontent.com/stefan-niedermann/paypal-donate-button/master/paypal-donate-button.png)](https://www.paypal.com/donate?hosted_button_id=9ELH753DDE98Y)
        
