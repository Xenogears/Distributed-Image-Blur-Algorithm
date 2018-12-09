#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define MAXCAD 100

#define DEFAULT_NUM_PASOS 	1
#define DEFAULT_DIST_RADIO 	5

#define ERROR_EXIT_F(A, B) { fprintf(stderr, A, B); return -1; }
#define ERROR_EXIT(A) { fprintf(stderr, A); return -1; }

#define STEP_GATHERV	2
#define STEP_SENDRECV	1

#define INFO
//#define DEBUG
//#define DEBUG_EXTRA
//#define FAKE_ROWS		23
//#define FAKE_COLS 	4

struct pixel
{
	unsigned char r, g, b;
};

int lee_ppm(char *nomfich, struct pixel ***img, int *nf, int *nc)
{
	FILE *df;
	char tipo[MAXCAD];
	int nivmax, height, width, i, j;

	df = fopen(nomfich, "r");
	if (!df) 
		return -1;   /* fopen ha fallado */

	fscanf(df, "%s", tipo);
	if (strcmp(tipo, "P3") != 0) 
		return -2;   /* formato erróneo */

	fscanf(df, "%d%d", &width, &height);

	if ((*img = (struct pixel **)malloc(sizeof(struct pixel *) * height)) == NULL) 
		return -3;   /* malloc ha fallado */
	
	if (((*img)[0] = (struct pixel *)malloc(sizeof(struct pixel) * width * height)) == NULL) 
		return -3;   /* malloc ha fallado */
	
	for (i = 1; i < height; i++)
		(*img)[i] = (*img)[i - 1] + width;

	fscanf(df, "%d", &nivmax);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
		  fscanf(df, "%hhu", &((*img)[i][j].r));
		  fscanf(df, "%hhu", &((*img)[i][j].g));
		  fscanf(df, "%hhu", &((*img)[i][j].b));
		}
	}

	*nf = height;
	*nc = width;

	fclose(df);

	return 0;
}


int escribe_ppm(char *nomfich, struct pixel **img, int nf, int nc)
{
	FILE *df;
	int nivmax = 255, i, j;

	df = fopen(nomfich, "w");
	if (!df) 
		return -1;   /* fopen ha fallado */

	fprintf(df, "P3\n");
	fprintf(df, "%d %d\n", nc, nf);
	fprintf(df, "%d\n", nivmax);

	for (i = 0; i < nf; i++)
	{
		for (j = 0; j < nc; j++)
		{
		  fprintf(df, "%d ", img[i][j].r);
		  fprintf(df, "%d ", img[i][j].g);
		  fprintf(df, "%d ", img[i][j].b);
		}
		
		fprintf(df, "\n");
	}

	fclose(df);

	return 0;
}

//Esta función aplica una vez el filtrado de la imagen.
//Se ha modificado respecto al código propuesto para el caso de estudio con la intención de extraer todas las comunicaciones y así
//simplificar la paralelización con MPI.
void Filtro(int radio, struct pixel **ppsImagenOrg, struct pixel **ppsImagenDst, int **ppdBloque, int n, int nMax, int nOffset, int m)
{
	int i, j, k, l, tot;
	struct { int r, g, b; } resultado;
	int v;
	
	for (i = nOffset; i < (n+nOffset); i++)
	{
		//Paralelización OPENMP por tareas
		#pragma omp task firstprivate(i) private(j, k, l, resultado, tot, v) shared(nOffset, n, nMax, m, radio, ppsImagenOrg, ppsImagenDst) 
		for (j = 0; j < m; j++)
		{
			resultado.r = 0;
			resultado.g = 0;
			resultado.b = 0;
			tot = 0;
			
			for (k = max(0, i - radio); k <= min(nMax - 1, i + radio); k++)
			{
				for (l = max(0, j - radio); l <= min(m - 1, j + radio); l++)
				{					
					v = ppdBloque[k - i + radio][l - j + radio];
					#ifdef DEBUG_EXTRA						
						printf("Pos[i=%d][j=%d] -> k=%d, l=%d --> ppdBloque[%d][%d] = %d\n", i, j, k, l, (k - i + radio), (l - j + radio), v);
					#endif
					
					resultado.r += ppsImagenOrg[k][l].r * v;
					resultado.g += ppsImagenOrg[k][l].g * v;
					resultado.b += ppsImagenOrg[k][l].b * v;
					tot += v;
				}
			}
			
			resultado.r /= tot;
			resultado.g /= tot;
			resultado.b /= tot;
			ppsImagenDst[i-nOffset][j].r = resultado.r;
			ppsImagenDst[i-nOffset][j].g = resultado.g;
			ppsImagenDst[i-nOffset][j].b = resultado.b;	
		}
	}
	
	#pragma omp taskwait
	return;
}

//Función auxiliar para mostrar un vector.
void printVector(char *name, int *v, unsigned int n)
{
	unsigned int i = 0;
	
	printf("%s = [", name);
	for(i = 0; i < n; i++)
	{
		printf("%d", v[i]);
		if(i < (n-1))
			printf(", ");				
	}
	printf("]\n");		
}

//Esta función reserva y crea la matriz con los pesos para aplicar el filtro.
int createPpdBloque(int ***ppdBloque, int radio)
{
	int i, j;
	
	if ((*ppdBloque = (int **)malloc(sizeof(int *) * (2*radio + 1))) == NULL)
		return 0;   /* malloc ha fallado */

	if (((*ppdBloque)[0] = (int *)malloc(sizeof(int) * (2*radio + 1) * (2*radio + 1))) == NULL)
		return 0;   /* malloc ha fallado */  

	for (i = 1; i < 2*radio + 1; i++)
		(*ppdBloque)[i] = (*ppdBloque)[i - 1] + 2*radio + 1;

	for (i = -radio; i <= radio; i++)
		for (j = -radio; j <= radio; j++)
			(*ppdBloque)[i + radio][j + radio] = (radio - abs(i)) * (radio - abs(i)) + (radio - abs(j)) * (radio - abs(j)) + 1;	
		
	return 1;
}

//Esta función se encarga de reservar los buffers en memoria con los tamaños adecuados para cada proceso.
int allocateImageBuffers(struct pixel ***ImgOrg, struct pixel ***ImgDst, int rank, int rowsCountOrg, int rowsCountDst, int colsCount)
{
	int i;
	
	//IMG SRC
	if(rank > 0)
	{		
		if ((*ImgOrg = (struct pixel**)malloc(sizeof(struct pixel *) * rowsCountOrg)) == NULL) 
			return 0;   /* malloc ha fallado */
		
		if (((*ImgOrg)[0] = (struct pixel*)malloc(sizeof(struct pixel) * rowsCountOrg * colsCount)) == NULL) 
			return 0;   /* malloc ha fallado */
		
		for (i = 1; i < rowsCountOrg; i++)
			(*ImgOrg)[i] = (*ImgOrg)[i - 1] + colsCount;
		
		#ifdef DEBUG
			memset(ImgOrg[0], (sizeof(struct pixel) * rowsCountOrg * colsCount), 0);
		#endif
	}
	
	//IMG DST
	if ((*ImgDst = (struct pixel**)malloc(sizeof(struct pixel *) * rowsCountDst)) == NULL) 
		return 0;   /* malloc ha fallado */
	
	if (((*ImgDst)[0] = (struct pixel*)malloc(sizeof(struct pixel) * rowsCountDst * colsCount)) == NULL) 
		return 0;   /* malloc ha fallado */
	
	for (i = 1; i < rowsCountDst; i++)
		(*ImgDst)[i] = (*ImgDst)[i - 1] + colsCount;
	
	#ifdef DEBUG
		memset(ImgDst[0], (sizeof(struct pixel) * rowsCountDst * colsCount), 0);
	#endif
	
	return 1; //OK
}

int main(int argc, char *argv[])
{
	struct pixel **ImgOrg, **ImgDst;
	int **ppdBloque;
	unsigned int rowsCount, colsCount;
	int i, j, rc;	
	double prevTime, postTime;
	int stepMode = STEP_SENDRECV;
	
	int *counts, *countsB, *offsets, *offsetsB;
	int radius = DEFAULT_DIST_RADIO; 
	int steps = DEFAULT_NUM_PASOS;
	
	int rank, size;
	MPI_Datatype rowType, pxType;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);	
	
	int blockLen[] = { 3 };
	MPI_Aint indices[] = { offsetof(struct pixel, r) };
	MPI_Datatype types[] = { MPI_CHAR };
	MPI_Type_create_struct(1, blockLen, indices, types, &pxType);
	MPI_Type_commit(&pxType);

	if(argc < 3)
		ERROR_EXIT_F("Use: %s srcImg destImg [radius] [steps] [sendmode between steps (1=Gatherv, 2=SendRecv)]\n", argv[0]);		
	
	if(argc >= 4)
		sscanf(argv[3], "%d", &radius);	

	if(argc >= 5)
		sscanf(argv[4], "%d", &steps);	
		
	if(argc >= 6)
		sscanf(argv[5], "%d", &stepMode);
	
	if(rank == 0)
	{
		rc = lee_ppm(argv[1], &ImgOrg, &rowsCount, &colsCount);
		if (rc)
			ERROR_EXIT_F("Error al leer el fichero %s\n", argv[1]);
		
		radius = min(radius, rowsCount);
		
		#ifdef FAKE_ROWS
			rowsCount = FAKE_ROWS;
			#endif
		#ifdef FAKE_COLS
			colsCount = FAKE_COLS;
		#endif		
		
		#ifdef INFO
			printf("Abierta una imagen de %d filas y %d columnas.\n", rowsCount, colsCount);			
			printf("Radio %d y %d pasos.\n", radius, steps);
			printf("MPI: %d procesos\n", size);
			#ifdef _OPENMP
				printf("OPENMP: %d hilos\n", omp_get_max_threads());
			#endif
			if(stepMode == STEP_GATHERV)
				printf("Usando Gatherv/Scatterv entre pasos\n");
			else
				printf("Usando Isend/Irecv entre pasos\n");
			
			fflush(stdout);
		#endif
		
		prevTime = MPI_Wtime();	
	}
	
	//Enviamos a todos los procesos el número de filas y columnas
	MPI_Bcast(&rowsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	//Creamos la matriz PPD Bloque 
	//(Anteriormente se encontraba en la función Filtro, pero debido al cambio de la estructura del programa, procede sacarlo a aquí para evitar tareas repetidas)
	if(!createPpdBloque(&ppdBloque, radius))
		ERROR_EXIT("Error creating ppdBloque");
	
	counts 	 = (int*)malloc(size * sizeof(int));
	countsB  = (int*)malloc(size * sizeof(int));
	offsets  = (int*)malloc(size * sizeof(int));
	offsetsB = (int*)malloc(size * sizeof(int));
	unsigned int tempCounts = (rowsCount / size);
	unsigned int remCounts = (rowsCount % size);
	
	//Este reparto tiene como objetivo distribuir de la forma más equilibrada posible el número de filas entre los distintos procesos.
	//Además evita en la medida de lo posible aplicar menos carga al proceso con rango 0, el cual generalmente se encarga mayormente de las comunicaciones y por ello tiene mayor carga.
	for(i = 0; i < size; i++)
	{
		counts[i] = tempCounts;
		
		if(remCounts > 0 && i != 0)  
		{
			counts[i]++;
			remCounts--;
		}
		
		if(i > 0)		
			offsets[i] = (offsets[i-1] + counts[i-1]);		
		else 
			offsets[i] = 0;
	}
	
	for(i = 0; i < size; i++)
	{					
		countsB[i] = counts[i] + min(radius, offsets[i]) + min(radius, rowsCount-offsets[i]);		
		offsetsB[i] = max(offsets[i] - radius, 0);
		
		if(countsB[i] + offsetsB[i] > rowsCount)
			countsB[i] = (rowsCount - offsetsB[i]);
	}
	
	#ifdef DEBUG
		if(rank == 0)
		{						
			printVector("counts", counts, size);
			printVector("countsB", countsB, size);			
			printVector("offsets", offsets, size);
			printVector("offsetsB", offsetsB, size);
						
			#ifdef DEBUG_EXTRA
				char name[64];		
				int s = (2*radius + 1);
				for(i = 0; i < s; i++)
				{
					sprintf(name, "ppdBloque[%d]", i);
					printVector(name, ppdBloque[i], s);
				}
			#endif
		}
	#endif	
	
	MPI_Type_vector(1, colsCount, 1, pxType, &rowType);
	MPI_Type_commit(&rowType);
	
	//Reserva de los buffers de memoria para ImgOrg e ImgDst
	if(!allocateImageBuffers(&ImgOrg, &ImgDst, rank, countsB[rank], (rank == 0 ? rowsCount : counts[rank]), colsCount))	
		ERROR_EXIT("Error allocating image buffers\n");	
	
	//Reparto inicial
	MPI_Scatterv(ImgOrg[0], countsB, offsetsB, rowType, ImgOrg[0], countsB[rank], rowType, 0, MPI_COMM_WORLD);
			
	//Desplazamiento donde comienza la fila a cacular en Filtro
	int rowOffset = (offsets[rank] - offsetsB[rank]);	
	for(i = 0; i < steps; i++)
	{
		#pragma omp parallel
		#pragma omp single
		Filtro(radius, ImgOrg, ImgDst, ppdBloque, counts[rank], countsB[rank], rowOffset, colsCount);

		if(i < (steps-1)) //Si hay un paso siguiente
		{
			if(stepMode == STEP_GATHERV)
			{
				//Solución 1 para la comunicación entre pasos: GatherV y ScatterV.
				//Esta solución es la más sencilla a nivel de código pero añade una sobrecarga en las comunicaciones entre pasos.
				//Se envía el resultado parcial de vuelta al proceso con rango 0 para que este lo vuelva a distribuir entre los procesos como en el reparto inicial.
				//Esto resulta en un proceso ineficiente, ya que genera una sobrecarga sobre el proceso 0 y además envía en bloque más filas de las necesarias para continuar
				//el cómputo entre pasos.
				
				MPI_Gatherv(ImgDst[0], counts[rank], rowType, ImgDst[0], counts, offsets, rowType, 0, MPI_COMM_WORLD);
				MPI_Scatterv(ImgDst[0], countsB, offsetsB, rowType, ImgOrg[0], countsB[rank], rowType, 0, MPI_COMM_WORLD);
			}
			else if(stepMode == STEP_SENDRECV)
			{	
				//Solución 2 para la comunicación entre pasos: Isend y Irecv
				//Esta solución trata de resolver los problemas de la solución anterior.
				//Cada proceso envía a sus procesos adyacentes el número de filas necesario para continuar con el Filtrado de la imagen.
				//De esta forma, no sólo se evita sobrecargar al proceso 0, sino que además se reduce el número de datos a enviar a sólo los necesarios.
				//Se emplean comunicaciones no bloqueantes para evitar un orden de envío y recepción entre los procesos (además de posibles interbloqueos).
				
				int myStartRow = offsets[rank];
				int myEndRow = (myStartRow + counts[rank] - 1);
				int myStartRowB = offsetsB[rank];
				int myEndRowB = (myStartRowB + countsB[rank] - 1);				
				
				MPI_Request recvReq[size];
				MPI_Request sendReq[size];
				int recvReqIndex = 0;
				int sendReqIndex = 0;
								
				for(j = 0; j < size; j++)
				{					
					int pStartRow = offsets[j];
					int pEndRow = (pStartRow + counts[j] - 1);
					int pStartRowB = offsetsB[j];
					int pEndRowB = (pStartRowB + countsB[j] - 1);
					
					if(j < rank) //Envío y recepción a los procesos con un rango menor
					{
						int rowsToRecv = min(pEndRow - myStartRowB + 1, counts[j]);
						int rowsToSend = min(pEndRowB - myStartRow + 1, counts[rank]);

						if(rowsToRecv > 0 || rowsToSend > 0)
						{
							#ifdef DEBUG				
								printf("P%d -> GOING TO RECV %d ROWS IN ImgOrg[%d] FROM %d PROC (LOWER)\n", rank, rowsToRecv, max(offsets[j] - offsetsB[rank], 0), j);
								printf("P%d -> GOING TO SEND %d ROWS FROM ImgDst[0] TO %d PROC (LOWER)\n", rank, rowsToSend, j);
							#endif
							
							MPI_Isend(ImgDst[0], rowsToSend, rowType, j, 1, MPI_COMM_WORLD, &sendReq[sendReqIndex++]); 
							MPI_Irecv(ImgOrg[max(offsets[j] - offsetsB[rank], 0)], rowsToRecv, rowType, j, 1, MPI_COMM_WORLD, &recvReq[recvReqIndex++]);							
						}				
					}
					else if(j > rank) //Envío y recepción a los procesos con un rango mayor
					{
						int rowsToRecv = min(myEndRowB - pStartRow + 1, counts[j]);
						int rowsToSend = min(myEndRow - pStartRowB + 1, counts[rank]);	
						
						if(rowsToRecv > 0 || rowsToSend > 0)
						{
							#ifdef DEBUG
								printf("P%d -> GOING TO RECV %d ROWS IN ImgOrg[%d] FROM %d PROC (HIGHER)\n", rank, rowsToRecv, offsets[j] - offsetsB[rank], j);
								printf("P%d -> GOING TO SEND %d ROWS FROM ImgDst[%d] TO %d PROC (HIGHER)\n", rank, rowsToSend, counts[rank]-1, j);
							#endif
							
							MPI_Isend(ImgDst[counts[rank] - rowsToSend], rowsToSend, rowType, j, 1, MPI_COMM_WORLD, &sendReq[sendReqIndex++]); 
							MPI_Irecv(ImgOrg[offsets[j] - offsetsB[rank]], rowsToRecv, rowType, j, 1, MPI_COMM_WORLD, &recvReq[recvReqIndex++]);
						}
					}
				}	

				//Copiamos la parte que corresponde a la sección del proceso actual en ImgOrg para que esté disponible en el próximo paso.
				//Este paso se realiza aquí para aprovechar el retardo de la copia para que las comunicaciones se completen
				memcpy(ImgOrg[rowOffset], ImgDst[0], (counts[rank] * colsCount * sizeof(struct pixel)));
		
				//Esperamos a que todas las comunicaciones de envío y recepción se completen.
				MPI_Waitall(sendReqIndex, sendReq, MPI_STATUSES_IGNORE);
				MPI_Waitall(recvReqIndex, recvReq, MPI_STATUSES_IGNORE);
			}
		}
	}	
		
	//Obtenemos los resultados de todos los procesos
	MPI_Gatherv(ImgDst[0], counts[rank], rowType, ImgDst[0], counts, offsets, rowType, 0, MPI_COMM_WORLD);	
	
	if(rank == 0)
	{
		postTime = MPI_Wtime();
		printf("Tiempo total: %f\n", (postTime - prevTime));
		
		rc = escribe_ppm(argv[2], ImgDst, rowsCount, colsCount);		
		if (rc)
			ERROR_EXIT_F("Error al escribir la imagen en el fichero %s\n", argv[2]);
	}
	
	free(ppdBloque[0]);
	free(ppdBloque);
	free(ImgOrg[0]);
	free(ImgDst[0]);
	free(ImgOrg);
	free(ImgDst);	
	free(counts);
	free(countsB);
	free(offsets);
	free(offsetsB);

	MPI_Type_free(&pxType);
	MPI_Type_free(&rowType);
	MPI_Finalize();	
	
	return 0;
}
