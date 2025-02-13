from functools import reduce

def bicluster_fitness_value(c_rows, c_columns, final_pivot_matrix):
    rows_means = []
    for k, cluster in enumerate(c_rows):
      row_mean = []
      for i, row in enumerate(cluster):
        if row == True:
          sum = 0
          num = 0
          for j, col in enumerate(c_columns[k]):
            if col == True:
              sum = sum + final_pivot_matrix[i][j]
              num = num + 1
          row_mean.append(sum/num)
      rows_means.append(row_mean)

    columns_means = []
    for k, cluster in enumerate(c_columns):
      column_mean = []
      for i, column in enumerate(cluster):
        if column == True:
          sum = 0
          num = 0
          for j, row in enumerate(c_rows[k]):
            if row == True:
              sum = sum + final_pivot_matrix[i][j]
              num = num + 1
          column_mean.append(sum/num)
      columns_means.append(column_mean)
      
    matrices_means = []
    for i, cluster in enumerate(rows_means):
      num = len(cluster) * len(columns_means[i])
      sum = reduce(lambda x, y: x + y, cluster)
      matrices_means.append(sum/num)
      
    clusters_residues = []
    for k, cluster in enumerate(c_rows):
      x = 0
      residues = []
      for i, row in enumerate(cluster):
        if row == True:
          row_residues = []
          y = 0
          for j, col in enumerate(c_columns[k]):
            if col == True:
              residue = final_pivot_matrix[i][j] - columns_means[k][y] - rows_means[k][x] + matrices_means[k]
              row_residues.append(residue)
              y = y + 1
          x = x + 1
        residues.append(row_residues)
      clusters_residues.append(residues)
    
    del matrices_means
    del rows_means
    del columns_means
    MSRs = []
    Fitness_Values = []
    for i, cluster_residue in enumerate(clusters_residues):
      I = len(cluster_residue)
      J = len(cluster_residue[0])
      size = I*J
      flat_list = [item for sublist in cluster_residue for item in sublist]
      sum_of_squares = reduce(lambda acc, x: acc + x**2, flat_list, 0)
      msr = sum_of_squares/size
      MSRs.append(msr)
      Fitness_Values.append(msr + (1/I) + (1/J))
      
    return Fitness_Values