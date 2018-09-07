# EJERCICIO 1
primer_w= function(w)
{ 
  letra_azar=sample(c(0,1),1,replace= FALSE)
  mi_palabra=seq(1,length(w),1)
  contador=0
  longitud=length(w)
  cond=FALSE
  while (cond==FALSE)
  {
    letra_azar=sample(c(0,1),1,replace = FALSE)

    for (i in 1:(longitud-1))
    {
      mi_palabra[i]=mi_palabra[i+1]
    }
    mi_palabra[longitud]=letra_azar
    contador=contador+1

    cond=identical(mi_palabra,w)

  }
  return (contador)
}
# EJERCICIO 2
tiempo_esperado=function(w,repeticiones)
{
  esperanza=0
  for (i in 1:repeticiones)
  {
    esperanza=esperanza+primer_w(w)
  }
  return (esperanza/repeticiones)
    
}

proba_w=function(w,k,repeticiones)
{
  cantidad=0
  for (i in 1:repeticiones)
  {
    if(primer_w(w)==k)
    {
      cantidad=cantidad+1  
    }
  }
  return (cantidad/repeticiones)
}

AA=c(0,0)
AB=c(0,1)
AAA=c(0,0,0)
AAB=c(0,0,1)
BBA=c(1,1,0)
ABA=c(0,1,0)
BAB=c(1,0,1)
BABA=c(1,0,1,0)
BABAB=c(1,0,1,0,1)
tiempo_esperado(AA,1000)
tiempo_esperado(AB,1000)
tiempo_esperado(AAA,1000)
tiempo_esperado(AAB,1000)
tiempo_esperado(BBA,1000)
tiempo_esperado(ABA,1000)
tiempo_esperado(BAB,1000)
tiempo_esperado(BABA,1000)
tiempo_esperado(BABAB,1000)

#comparo AA con AB
proba_w(AA,3,1000)
proba_w(AB,3,1000)


# EJERCICIO 3
# REALIZO UN HISTOGRAMA PARA OBSERVAR LA DISTRIBUCION DE LA VARIABLE ALEATORIA

plot_hist=function(w,repeticiones) 
{
  primeros_tiempos=seq(1,repeticiones,1)
  for (i in 1:repeticiones)
  {
    primeros_tiempos[i]=primer_w(w)
  }
  hist(primeros_tiempos)
}

plot_hist(AA,1000)

#comparo el resultado el histograma con una geometrica de parametro p=0.125

geometrica=rgeom(1000,0.125)
hist(geometrica)

# NO PUEDE CONSIDERARSE QUE LA DISTRIBUCION ES NORMAL, ES SIMILAR A UNA DISTRIBUCION GEOMETRICA.



