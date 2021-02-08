"""
Code for reading GRB data from Swift website
"""
import os
import sys
import time
import urllib
import urllib.request

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

dirname = os.path.dirname(__file__)


def get_trigger_number(grb):
    data = get_grb_table()
    trigger = data.query('GRB == @GRB')['Trigger Number']
    trigger = trigger.values[0]
    return trigger


def get_grb_table():
    short_table = os.path.join(dirname, 'tables/SGRB_table.txt')
    sgrb = pd.read_csv(short_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    long_table = os.path.join(dirname, 'tables/LGRB_table.txt')
    lgrb = pd.read_csv(long_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    frames = [lgrb, sgrb]
    data = pd.concat(frames, ignore_index=True)
    return data


def check_element(driver, id_number):
    """
    checks that an element exists on a website, and provides an exception
    """
    try:
        driver.find_element_by_id(id_number)
    except NoSuchElementException:
        return False
    return True


def get_grb_file(grb, use_default_directory):
    """
    Go to Swift website and get the data for a given GRB
    """

    if use_default_directory:
        grb_dir = os.path.join(dirname, '../data/GRBData/GRB' + grb + '/')
    else:
        grb_dir = 'GRBData/GRB' + grb + '/'

    # check if data file exists for this GRB:
    grb_datfile = grb_dir + 'GRB' + grb + '_rawSwiftData.dat'
    if not os.path.exists(grb_dir):
        os.makedirs(grb_dir)
    if os.path.isfile(grb_datfile):
        print('The raw data file already exists')
        # check = input('Do you still want to download fresh data? (y/[n])') or 'n'
        # if check == 'n':
        #     print('Exiting from getGRBFile function')
        return None

    # open the webdriver
    driver = webdriver.PhantomJS('/Users/nsarin/Documents/PhD/phantomjs-2.1.1-macosx/bin/phantomjs')

    # get the trigger number
    # print('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    # print('opening Swift website for GRB' + GRB)
    driver.get(grb_website)

    # celect option for BAT binning
    bat_binning = 'batxrtbin'
    if check_element(driver, bat_binning):
        driver.find_element_by_xpath("//select[@name='batxrtbin']/option[text()='SNR 4']").click()

    # select option for subplot
    subplot = "batxrtsub"
    if check_element(driver, subplot):
        driver.find_element_by_xpath("//select[@name='batxrtsub']/option[text()='no']").click()

    # Select option for flux density
    flux_density1 = "batxrtband1"
    flux_density0 = "batxrtband0"
    if (check_element(driver, flux_density1)) and (check_element(driver, flux_density0)):
        driver.find_element_by_xpath(".//*[@id='batxrtband1']").click()
        driver.find_element_by_xpath(".//*[@id='batxrtband0']").click()

    # Generate data file
    try:
        driver.find_element_by_xpath(".//*[@id='batxrt_XRTBAND_makeDownload']").click()
        time.sleep(20)

        grb_url = driver.current_url

        # Close the driver and all opened windows
        driver.quit()

        # scrape the data
        urllib.request.urlretrieve(grb_url, grb_datfile)
    except Exception:
        print('cannot load the website for GRB', grb)


def sort_data(grb, use_default_directory):
    if use_default_directory:
        grb_dir = os.path.join(dirname, '../data/GRBData/GRB' + grb + '/')
    else:
        grb_dir = 'GRBData/GRB' + grb + '/'

    raw_grb_datfile = grb_dir + 'GRB' + grb + '_rawSwiftData.dat'

    grb_outfile = grb_dir + 'GRB' + grb + '.dat'

    if os.path.isfile(grb_outfile):
        print('Processed data file already exists for GRB', grb)
        return None
    if not os.path.isfile(raw_grb_datfile):
        print('There is no raw data for GRB', grb)
        return None

    xrtpcflag = 0
    xrtpc = []

    batflag = 0
    bat = []

    xrtwtflag = 0
    xrtwt = []

    try:
        with open(raw_grb_datfile) as data:
            for num, line in enumerate(data):
                if line[0] == '!':
                    if line[2:13] == 'batSNR4flux':
                        xrtpcflag = 0
                        batflag = 1
                        xrtwtflag = 0
                    elif line[2:11] == 'xrtpcflux':
                        xrtpcflag = 1
                        batflag = 0
                        xrtwtflag = 0
                    elif line[2:11] == 'xrtwtflux':
                        xrtpcflag = 0
                        batflag = 0
                        xrtwtflag = 1
                    else:
                        xrtpcflag = 0
                        batflag = 0
                        xrtwtflag = 0

                if xrtpcflag == 1:
                    xrtpc.append(line)
                if batflag == 1:
                    bat.append(line)
                if xrtwtflag == 1:
                    xrtwt.append(line)

        with open(grb_outfile, 'w') as out:
            out.write('## BAT - batSNR4flux\n')
            for ii in range(len(bat)):
                try:
                    int(bat[ii][0])
                    out.write(bat[ii])
                except ValueError:
                    pass

            out.write('\n')
            out.write('## XRT - xrtwtflux\n')
            for ii in range(len(xrtwt)):
                try:
                    int(xrtwt[ii][0])
                    out.write(xrtwt[ii])
                except ValueError:
                    pass

            out.write('\n')
            out.write('## XRT - xrtpcflux\n')

            for ii in range(len(xrtpc)):
                try:
                    int(xrtpc[ii][0])
                    out.write(xrtpc[ii])
                except ValueError:
                    pass
    except IOError:
        try:
            print('There was an error opening the file')
            sys.exit()
        except SystemExit:
            pass
    print('congratulations, you now have a nice data file: ' + grb_outfile)


def retrieve_and_process_data(grb, use_default_directory=False):
    if use_default_directory:
        grb_dir = os.path.join(dirname, '../data/GRBData/GRB' + grb + '/')
    else:
        grb_dir = 'GRBData/GRB' + grb + '/'

    # check if data file exists for this GRB:
    grb_datfile = grb_dir + 'GRB' + grb + '.dat'

    if os.path.isfile(grb_datfile):
        print('The data file already exists for GRB', grb)
        return None
    else:
        get_grb_file(grb=grb, use_default_directory=use_default_directory)
        sort_data(grb=grb, use_default_directory=use_default_directory)