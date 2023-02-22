import xlsxwriter
import numpy as np

class DataWriter():

	def __init__(self, fname, col_name_="noise", row_name_="rho"):
		self.workbook = xlsxwriter.Workbook(fname)
		self.worksheet = self.workbook.add_worksheet()
		self.worksheets = {}
		self.col_name = col_name_
		self.row_name = row_name_
		self.row_fill = ""
		self.col_fill = ""

	def set_row_fill(self, tmp):
		self.row_fill = tmp

	def set_col_fill(self, tmp):
		self.col_fill = tmp

	def write_log(self, data, row_, col_, title, data_name_, run_):

		if ("Run %d" % run_) in self.worksheets:
			tmp_work_sheet = self.worksheets["Run %d" % run_]
		else:
			tmp_work_sheet = self.workbook.add_worksheet("Run %d" % run_)
			self.worksheets["Run %d" % run_] = tmp_work_sheet
			
		data_name = data_name_ + " %d" % run_

		noises, rhos = title
		row = row_
		col = col_
		tmp_work_sheet.write(row, col, data_name)

		row += 1
		col += 1
		for rho in rhos:
			tmp_work_sheet.write(row, col, rho)
			col += 1
		tmp_work_sheet.write(row, col, self.row_name)
		
		row += 1
		i = 0
		for noise in noises:
			col = col_
			tmp_work_sheet.write(row, col, noise)
			for rho in rhos:
				col += 1
				tmp = data[i][col - col_ - 1]
				if np.isinf(tmp):
					tmp = "inf"
				elif np.isnan(tmp):
					tmp = "nan"
				tmp_work_sheet.write(row, col, tmp)
			row += 1
			i += 1
		tmp_work_sheet.write(row, col_, self.col_name)

	def write_block(self, data, row_, col_, title, data_name):
		
		noises, rhos = title
		row = row_
		col = col_
		self.worksheet.write(row, col, data_name)

		row += 1
		col += 1
		for rho in rhos:
			if self.row_fill == "":
				self.worksheet.write(row, col, rho)
			else:
				self.worksheet.write(row, col, self.row_fill)

			col += 1
		self.worksheet.write(row, col, self.row_name)
		
		row += 1
		i = 0
		for noise in noises:
			col = col_
			if self.col_fill == "":
				self.worksheet.write(row, col, noise)
			else:
				self.worksheet.write(row, col, self.col_fill)

			for rho in rhos:
				col += 1
				tmp = data[i][col - col_ - 1]
				if np.isinf(tmp):
					tmp = "inf"
				elif np.isnan(tmp):
					tmp = "nan"
				self.worksheet.write(row, col, tmp)
			row += 1
			i += 1
		self.worksheet.write(row, col_, self.col_name)

	def __del__(self):
		self.workbook.close()