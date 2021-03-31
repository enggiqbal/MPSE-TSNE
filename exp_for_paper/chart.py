import pandas as pd
import matplotlib.pyplot as plt
def perspectives2():
    df=pd.read_csv("perspectives2.csv")

    df=df[['perplexity'  ,'n_perspectives',   'separation_error', 'time']]
    df=df.groupby(by=["perplexity","n_perspectives"]).mean().reset_index()

    # multiple line plots
    plt.plot( 'n_perspectives', 'separation_error', data=df[df['perplexity']==40], marker='x', label="Perplexity 40" )
    plt.plot( 'n_perspectives', 'separation_error', data=df[df['perplexity']==80],  marker='o' ,label="Perplexity 80")
    plt.plot( 'n_perspectives', 'separation_error', data=df[df['perplexity']==160], marker='*' ,  label="Perplexity 160")
    plt.plot( 'n_perspectives', 'separation_error', data=df[df['perplexity']==240],  marker='s',  label="Perplexity 240")
    plt.ylabel("Separation Error in %")
    plt.xlabel("# Perspectives")
    # show legend
    plt.legend()
    plt.savefig("perspectives2.png")
    # show graph
    # plt.show()

# perspectives2()

def two_clusters():
    df1=pd.read_csv("two_clusters_3_0.2.csv")
    df1=df1[["n_samples","time"]]
    df1=df1.groupby(by=["n_samples"]).mean().reset_index()


    df2=pd.read_csv("two_clusters_2_0.2.csv")
    df2=df2[["n_samples","time"]]
    df2=df2.groupby(by=["n_samples"]).mean().reset_index()

    plt.plot( 'n_samples', 'time', data=df1, marker='x', label="Perspectives #3" )
    plt.plot( 'n_samples', 'time', data=df2,  marker='o' ,label="Perspectives #2")
    plt.legend()

    plt.ylabel("Computation Time in Seconds")
    plt.xlabel("Data Points")
    plt.savefig("time_exp.png")
 
two_clusters()