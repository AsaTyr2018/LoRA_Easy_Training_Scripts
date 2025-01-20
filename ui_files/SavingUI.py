# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SavingUI.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QWidget)

from modules.DragDropLineEdit import DragDropLineEdit
from modules.LineEditHighlight import LineEditWithHighlight
from modules.ScrollOnSelect import (ComboBox, SpinBox)

class Ui_saving_ui(object):
    def setupUi(self, saving_ui):
        if not saving_ui.objectName():
            saving_ui.setObjectName(u"saving_ui")
        saving_ui.resize(515, 296)
        saving_ui.setMinimumSize(QSize(515, 0))
        self.gridLayout_2 = QGridLayout(saving_ui)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.formLayout_4 = QFormLayout()
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.save_state_enable = QCheckBox(saving_ui)
        self.save_state_enable.setObjectName(u"save_state_enable")

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.save_state_enable)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.save_last_state_enable = QCheckBox(saving_ui)
        self.save_last_state_enable.setObjectName(u"save_last_state_enable")
        self.save_last_state_enable.setEnabled(False)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_last_state_enable.sizePolicy().hasHeightForWidth())
        self.save_last_state_enable.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.save_last_state_enable)

        self.save_last_state_selector = ComboBox(saving_ui)
        self.save_last_state_selector.addItem("")
        self.save_last_state_selector.addItem("")
        self.save_last_state_selector.setObjectName(u"save_last_state_selector")
        self.save_last_state_selector.setEnabled(False)
        self.save_last_state_selector.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_4.addWidget(self.save_last_state_selector)

        self.save_last_state_input = SpinBox(saving_ui)
        self.save_last_state_input.setObjectName(u"save_last_state_input")
        self.save_last_state_input.setEnabled(False)
        self.save_last_state_input.setFocusPolicy(Qt.StrongFocus)
        self.save_last_state_input.setMinimum(1)
        self.save_last_state_input.setMaximum(16777215)

        self.horizontalLayout_4.addWidget(self.save_last_state_input)


        self.formLayout_4.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_4)


        self.gridLayout_2.addLayout(self.formLayout_4, 2, 0, 1, 2)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label = QLabel(saving_ui)
        self.label.setObjectName(u"label")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.output_folder_input = DragDropLineEdit(saving_ui)
        self.output_folder_input.setObjectName(u"output_folder_input")

        self.horizontalLayout.addWidget(self.output_folder_input)

        self.output_folder_selector = QPushButton(saving_ui)
        self.output_folder_selector.setObjectName(u"output_folder_selector")

        self.horizontalLayout.addWidget(self.output_folder_selector)


        self.formLayout_2.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout)

        self.output_name_enable = QCheckBox(saving_ui)
        self.output_name_enable.setObjectName(u"output_name_enable")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.output_name_enable)

        self.output_name_input = LineEditWithHighlight(saving_ui)
        self.output_name_input.setObjectName(u"output_name_input")
        self.output_name_input.setEnabled(False)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.output_name_input)

        self.easy_naming_label = QLabel(saving_ui)
        self.easy_naming_label.setObjectName(u"easy_naming_label")
        self.easy_naming_label.setMinimumSize(QSize(97, 0))

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.easy_naming_label)

        self.easy_naming_text_input = LineEditWithHighlight(saving_ui)
        self.easy_naming_text_input.setObjectName(u"easy_naming_text_input")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.easy_naming_text_input)


        self.gridLayout_2.addLayout(self.formLayout_2, 0, 0, 1, 2)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.resume_enable = QCheckBox(saving_ui)
        self.resume_enable.setObjectName(u"resume_enable")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.resume_enable)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.resume_input = DragDropLineEdit(saving_ui)
        self.resume_input.setObjectName(u"resume_input")
        self.resume_input.setEnabled(False)

        self.horizontalLayout_5.addWidget(self.resume_input)

        self.resume_selector = QPushButton(saving_ui)
        self.resume_selector.setObjectName(u"resume_selector")
        self.resume_selector.setEnabled(False)

        self.horizontalLayout_5.addWidget(self.resume_selector)


        self.formLayout.setLayout(0, QFormLayout.FieldRole, self.horizontalLayout_5)

        self.save_only_last_enable = QCheckBox(saving_ui)
        self.save_only_last_enable.setObjectName(u"save_only_last_enable")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.save_only_last_enable)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.save_last_selector = ComboBox(saving_ui)
        self.save_last_selector.addItem("")
        self.save_last_selector.addItem("")
        self.save_last_selector.setObjectName(u"save_last_selector")
        self.save_last_selector.setEnabled(False)
        self.save_last_selector.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_3.addWidget(self.save_last_selector)

        self.save_last_input = SpinBox(saving_ui)
        self.save_last_input.setObjectName(u"save_last_input")
        self.save_last_input.setEnabled(False)
        self.save_last_input.setFocusPolicy(Qt.StrongFocus)
        self.save_last_input.setMinimum(1)
        self.save_last_input.setMaximum(16777215)

        self.horizontalLayout_3.addWidget(self.save_last_input)


        self.formLayout.setLayout(1, QFormLayout.FieldRole, self.horizontalLayout_3)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.save_tag_input = DragDropLineEdit(saving_ui)
        self.save_tag_input.setObjectName(u"save_tag_input")
        self.save_tag_input.setEnabled(False)

        self.horizontalLayout_6.addWidget(self.save_tag_input)

        self.save_tag_selector = QPushButton(saving_ui)
        self.save_tag_selector.setObjectName(u"save_tag_selector")
        self.save_tag_selector.setEnabled(False)

        self.horizontalLayout_6.addWidget(self.save_tag_selector)


        self.formLayout.setLayout(2, QFormLayout.FieldRole, self.horizontalLayout_6)

        self.save_toml_enable = QCheckBox(saving_ui)
        self.save_toml_enable.setObjectName(u"save_toml_enable")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.save_toml_enable)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.save_toml_input = DragDropLineEdit(saving_ui)
        self.save_toml_input.setObjectName(u"save_toml_input")
        self.save_toml_input.setEnabled(False)

        self.horizontalLayout_7.addWidget(self.save_toml_input)

        self.save_toml_selector = QPushButton(saving_ui)
        self.save_toml_selector.setObjectName(u"save_toml_selector")
        self.save_toml_selector.setEnabled(False)

        self.horizontalLayout_7.addWidget(self.save_toml_selector)


        self.formLayout.setLayout(3, QFormLayout.FieldRole, self.horizontalLayout_7)

        self.save_tag_enable = QCheckBox(saving_ui)
        self.save_tag_enable.setObjectName(u"save_tag_enable")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.save_tag_enable)


        self.gridLayout_2.addLayout(self.formLayout, 1, 1, 1, 1)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_2 = QLabel(saving_ui)
        self.label_2.setObjectName(u"label_2")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_2)

        self.save_precision_selector = ComboBox(saving_ui)
        self.save_precision_selector.addItem("")
        self.save_precision_selector.addItem("")
        self.save_precision_selector.addItem("")
        self.save_precision_selector.setObjectName(u"save_precision_selector")
        self.save_precision_selector.setFocusPolicy(Qt.StrongFocus)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.save_precision_selector)

        self.label_3 = QLabel(saving_ui)
        self.label_3.setObjectName(u"label_3")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_3)

        self.save_as_selector = ComboBox(saving_ui)
        self.save_as_selector.addItem("")
        self.save_as_selector.addItem("")
        self.save_as_selector.addItem("")
        self.save_as_selector.setObjectName(u"save_as_selector")
        self.save_as_selector.setFocusPolicy(Qt.StrongFocus)

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.save_as_selector)

        self.save_ratio_enable = QCheckBox(saving_ui)
        self.save_ratio_enable.setObjectName(u"save_ratio_enable")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.save_ratio_enable)

        self.save_ratio_input = SpinBox(saving_ui)
        self.save_ratio_input.setObjectName(u"save_ratio_input")
        self.save_ratio_input.setEnabled(False)
        self.save_ratio_input.setFocusPolicy(Qt.StrongFocus)
        self.save_ratio_input.setMinimum(1)
        self.save_ratio_input.setValue(1)

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.save_ratio_input)

        self.save_freq_enable = QCheckBox(saving_ui)
        self.save_freq_enable.setObjectName(u"save_freq_enable")

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.save_freq_enable)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.save_freq_selector = ComboBox(saving_ui)
        self.save_freq_selector.addItem("")
        self.save_freq_selector.addItem("")
        self.save_freq_selector.setObjectName(u"save_freq_selector")
        self.save_freq_selector.setEnabled(False)
        self.save_freq_selector.setFocusPolicy(Qt.StrongFocus)

        self.horizontalLayout_2.addWidget(self.save_freq_selector)

        self.save_freq_input = SpinBox(saving_ui)
        self.save_freq_input.setObjectName(u"save_freq_input")
        self.save_freq_input.setEnabled(False)
        self.save_freq_input.setFocusPolicy(Qt.StrongFocus)
        self.save_freq_input.setMinimum(1)
        self.save_freq_input.setMaximum(16777215)

        self.horizontalLayout_2.addWidget(self.save_freq_input)


        self.formLayout_3.setLayout(3, QFormLayout.FieldRole, self.horizontalLayout_2)


        self.gridLayout_2.addLayout(self.formLayout_3, 1, 0, 1, 1)


        self.retranslateUi(saving_ui)

        QMetaObject.connectSlotsByName(saving_ui)
    # setupUi

    def retranslateUi(self, saving_ui):
        saving_ui.setWindowTitle(QCoreApplication.translate("saving_ui", u"Form", None))
#if QT_CONFIG(tooltip)
        self.save_state_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save State is how you save the training state so you can resume later</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_state_enable.setText(QCoreApplication.translate("saving_ui", u"Save State", None))
#if QT_CONFIG(tooltip)
        self.save_last_state_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Last State is much like Save Only Last in that it will keep only the last epoch or step number of states</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_last_state_enable.setText(QCoreApplication.translate("saving_ui", u"Save Last State", None))
        self.save_last_state_selector.setItemText(0, QCoreApplication.translate("saving_ui", u"Epochs", None))
        self.save_last_state_selector.setItemText(1, QCoreApplication.translate("saving_ui", u"Steps", None))

#if QT_CONFIG(tooltip)
        self.save_last_state_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Last State is much like Save Only Last in that it will keep only the last epoch or step number of states</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.save_last_state_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Last State is much like Save Only Last in that it will keep only the last epoch or step number of states</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.label.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Output Folder is the location the models are saved to</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("saving_ui", u"Output Folder", None))
#if QT_CONFIG(tooltip)
        self.output_folder_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Output Folder is the location the models are saved to</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.output_folder_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Output Folder", None))
#if QT_CONFIG(tooltip)
        self.output_folder_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Output Folder is the location the models are saved to</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.output_folder_selector.setText("")
#if QT_CONFIG(tooltip)
        self.output_name_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Output Name is the base name of the models that get saved</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.output_name_enable.setText(QCoreApplication.translate("saving_ui", u"Output Name", None))
#if QT_CONFIG(tooltip)
        self.output_name_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Output Name is the base name of the models that get saved</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.output_name_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Output Name", None))
#if QT_CONFIG(tooltip)
        self.easy_naming_label.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Easily rename all instances of ${baseName} variable in files/folders under Saving and Logging</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.easy_naming_label.setText(QCoreApplication.translate("saving_ui", u"Easy Naming", None))
#if QT_CONFIG(tooltip)
        self.easy_naming_text_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>What to replace all instances of ${baseName} with. To use, add ${baseName} in each desired textbox</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.easy_naming_text_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Base name for model", None))
#if QT_CONFIG(tooltip)
        self.resume_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Resume State resumes training at a previous training, assuming you save the state for that bake</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.resume_enable.setText(QCoreApplication.translate("saving_ui", u"Resume State", None))
#if QT_CONFIG(tooltip)
        self.resume_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Resume State resumes training at a previous training, assuming you save the state for that bake</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.resume_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Folder To Resume From", None))
#if QT_CONFIG(tooltip)
        self.resume_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Resume State resumes training at a previous training, assuming you save the state for that bake</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.resume_selector.setText("")
#if QT_CONFIG(tooltip)
        self.save_only_last_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Only Last is how you save models only after a specific epoch or step count</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_only_last_enable.setText(QCoreApplication.translate("saving_ui", u"Save Only Last", None))
        self.save_last_selector.setItemText(0, QCoreApplication.translate("saving_ui", u"Epochs", None))
        self.save_last_selector.setItemText(1, QCoreApplication.translate("saving_ui", u"Steps", None))

#if QT_CONFIG(tooltip)
        self.save_last_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Only Last is how you save models only after a specific epoch or step count</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.save_last_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Only Last is how you save models only after a specific epoch or step count</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.save_tag_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Tag File saves a txt file that contains a list of all tags within the dataset sorted by tag count. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_tag_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Folder to save to", None))
#if QT_CONFIG(tooltip)
        self.save_tag_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Tag File saves a txt file that contains a list of all tags within the dataset sorted by tag count. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_tag_selector.setText("")
#if QT_CONFIG(tooltip)
        self.save_toml_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Toml File saves a toml file that contains the configuration of the training. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_toml_enable.setText(QCoreApplication.translate("saving_ui", u"Save Toml File", None))
#if QT_CONFIG(tooltip)
        self.save_toml_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Toml File saves a toml file that contains the configuration of the training. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_toml_input.setPlaceholderText(QCoreApplication.translate("saving_ui", u"Folder to save to", None))
#if QT_CONFIG(tooltip)
        self.save_toml_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Toml File saves a toml file that contains the configuration of the training. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_toml_selector.setText("")
#if QT_CONFIG(tooltip)
        self.save_tag_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Tag File saves a txt file that contains a list of all tags within the dataset sorted by tag count. If enabled but no folder is provided, The default location will be in a folder called auto_save_store</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_tag_enable.setText(QCoreApplication.translate("saving_ui", u"Save Tag File", None))
#if QT_CONFIG(tooltip)
        self.label_2.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Precision is the precision the model is saved at, this doesn't neccessarily have to equal the training precision</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label_2.setText(QCoreApplication.translate("saving_ui", u"Save Precision", None))
        self.save_precision_selector.setItemText(0, QCoreApplication.translate("saving_ui", u"fp16", None))
        self.save_precision_selector.setItemText(1, QCoreApplication.translate("saving_ui", u"bf16", None))
        self.save_precision_selector.setItemText(2, QCoreApplication.translate("saving_ui", u"float", None))

#if QT_CONFIG(tooltip)
        self.save_precision_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Precision is the precision the model is saved at, this doesn't neccessarily have to equal the training precision</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.label_3.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save As is the file type you save the file as. Typically you'll want to save as safetensors</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label_3.setText(QCoreApplication.translate("saving_ui", u"Save As", None))
        self.save_as_selector.setItemText(0, QCoreApplication.translate("saving_ui", u"safetensors", None))
        self.save_as_selector.setItemText(1, QCoreApplication.translate("saving_ui", u"pt", None))
        self.save_as_selector.setItemText(2, QCoreApplication.translate("saving_ui", u"ckpt", None))

#if QT_CONFIG(tooltip)
        self.save_as_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save As is the file type you save the file as. Typically you'll want to save as safetensors</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.save_ratio_enable.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Ratio is the way to limit the number of models saved</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_ratio_enable.setText(QCoreApplication.translate("saving_ui", u"Save Ratio", None))
#if QT_CONFIG(tooltip)
        self.save_ratio_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>Save Ratio is the way to limit the number of models saved</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.save_freq_enable.setText(QCoreApplication.translate("saving_ui", u"Save Freq", None))
        self.save_freq_selector.setItemText(0, QCoreApplication.translate("saving_ui", u"Epochs", None))
        self.save_freq_selector.setItemText(1, QCoreApplication.translate("saving_ui", u"Steps", None))

#if QT_CONFIG(tooltip)
        self.save_freq_selector.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>How often to save models. You can save according to steps or epochs, setting it to epochs and 1 means it will save a model every epoch.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.save_freq_input.setToolTip(QCoreApplication.translate("saving_ui", u"<html><head/><body><p>How often to save models. You can save according to steps or epochs, setting it to epochs and 1 means it will save a model every epoch.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

