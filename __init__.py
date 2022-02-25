import argparse
import csv
import datetime as dt
import json
import logging
import os
import time
from io import StringIO
from types import prepare_class

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2
import requests
from psycopg2.extensions import AsIs, quote_ident
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_toolbelt import sessions

CFPB_URL = 'https://www.consumerfinance.gov'
DAYS_IN_MONTH = 31
DAYS_IN_YEAR = 365

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--debug',
    help='Print lots of debugging statements',
    action='store_const', dest='loglevel', const=logging.DEBUG,
    default=logging.WARNING,
)
parser.add_argument(
    '-v', '--verbose',
    help='Be verbose',
    action='store_const', dest='loglevel', const=logging.INFO,
)
parser.add_argument(
    '-u', '--user',
    help='Username for Postgres',
    dest='user',
    default='postgres'
)
parser.add_argument(
    '-pw', '--password',
    help='Password for Postgres',
    dest='password',
    default='admin'  # :)
)
parser.add_argument(
    '-db', '--database',
    help='Name of Postgres database',
    dest='database',
    default='postgres'
)
parser.add_argument(
    '-p', '--port',
    help='Port for connecting to Postgres',
    dest='port',
    default='5432'
)
parser.add_argument(
    '-t', '--table',
    help='Name of table',
    dest='tablename',
    default='cfpb_db'
)
parser.add_argument(
    '-g', '--graph',
    help='''Draw a graph counting updates each day from last month
            And save it to `count_from_%begin%_to_%end%_graph.png`''',
    action='store_true', dest='countgraph',
    default=False,
)
parser.add_argument(
    '-cc', '--compare',
    help='''Draw a graph comparing number of complaints for two different companies
            And save it to `compare_%company1%_and_%company2%_graph.png`''',
    nargs='+', dest='companies',
)


def json_to_dict(_id, **kwargs):
    return {'_id': _id} | kwargs


def json_to_df(json_data):
    return pd.DataFrame.from_records(
        [json_to_dict(i['_id'], **i['_source'])for i in json_data]
    )


def convert_to_date(x):
    return pd.to_datetime(x).dt.tz_localize(None)


def daterange(start_date, end_date, period=DAYS_IN_MONTH):
    for n in range(0, int((end_date - start_date).days), period):
        yield start_date + dt.timedelta(days=n)


def prepare_data(df):
    new_df = df.copy()
    new_df['_id'] = new_df['_id'].apply(pd.to_numeric, errors='coerce')
    new_df['update_stamp'] = dt.datetime.now()
    cols = new_df.columns.tolist()
    cols.remove('_id')
    date_cols = [i for i in cols if 'date' in i]
    text_cols = [i for i in cols if 'date' not in i]
    new_df = new_df.replace('\n', ' ', regex=True)
    new_df = new_df.replace(';', ',', regex=True)
    new_df = new_df.replace(r'\\', '', regex=True)
    new_df = new_df.replace('\"', '', regex=True)
    new_df = new_df.replace("\'", '', regex=True)
    new_df = new_df.replace('', np.NaN)
    new_df = new_df.replace(r'^\s*$', np.NaN, regex=True)
    new_df = new_df.replace('N/A', np.NaN)
    new_df = new_df.replace('None', np.NaN)
    new_df = new_df.fillna(value=np.NaN)
    new_df = new_df.applymap(lambda x: np.NaN
                             if isinstance(x, str) and (not x or x.isspace())
                             else x)
    new_df = new_df.replace({np.NaN: None})
    new_df[date_cols] = new_df[date_cols].apply(convert_to_date)
    new_df = new_df[['_id'] + date_cols + text_cols]
    return new_df


def connect_to_local_postgres(user, pw, db, port):
    return psycopg2.connect(
        host="localhost",
        database=db,
        user=user,
        password=pw,
        port=port,
    )


def check_table_exists(dbcon, tablename):
    with dbcon.cursor() as dbcur:
        dbcur.execute('''
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{0}'
            '''.format(tablename))
        return dbcur.fetchone()[0]


def get_session(session=None,
                base_url=None,
                retries=5,
                backoff_factor=0.5,
                raise_on_status=True,
                status_forcelist=(500, 502, 504)):
    session = session or sessions.BaseUrlSession(base_url=base_url or CFPB_URL)
    retry = Retry(total=retries,
                  backoff_factor=backoff_factor,
                  raise_on_status=raise_on_status,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def create_table(dbcon, name, values):
    update_stamp = 'update_stamp'
    data_val = [i for i in values if 'date' in i] + [update_stamp]
    text_val = [i for i in values if 'date' not in i]

    # remove old fields
    if 'has_narrative' in text_val:
        text_val.remove('has_narrative')

    with dbcon.cursor() as dbcur:
        dbcur.execute('CREATE TABLE {}\n'.format(name) +
                      '(_id INT NOT NULL,\n' +
                      '\n'.join(['{} TIMESTAMP,'.format(i) for i in data_val]) +
                      '\n'.join(['{} TEXT,'.format(i) for i in text_val]) +
                      'PRIMARY KEY(_id, {}));'.format(update_stamp))


def get_first_record(session):
    return download_chunk_from_api(sort='created_date_asc', size=1)


def get_data_for_all_time(dbcon,
                          session,
                          tablename,
                          starting_date,
                          period=DAYS_IN_YEAR):
    ending_date = dt.datetime.now()
    data = []
    for l_date in daterange(starting_date, ending_date, period):
        r_date = l_date + dt.timedelta(days=period)
        data = get_data_for_one_period(
            dbcon,
            session,
            tablename,
            l_date,
            r_date,
            True)
    return data


def get_data_for_one_period(dbcon,
                            session,
                            tablename,
                            starting_date=None,
                            ending_date=None,
                            ignore_previous=False):
    if starting_date and ending_date:
        min_date = starting_date
        max_date = ending_date
    elif starting_date:
        min_date = starting_date
        max_date = min_date + dt.timedelta(days=DAYS_IN_MONTH)
    else:
        max_date = dt.date.today()
        min_date = max_date - dt.timedelta(days=DAYS_IN_MONTH)
    max_date += dt.timedelta(days=1)

    df_data = json_to_df(download_chunk_from_api(session=session,
                                                 l_date=min_date,
                                                 r_date=max_date))
    logging.info('All data for period from {} to {} loaded'.format(min_date,
                                                                   max_date))
    df_data = prepare_data(df_data)
    return store_data(dbcon, df_data, tablename, ignore_previous, min_date, max_date)


def download_chunk_from_api(session=None,
                            timeout=100,
                            l_date=None,
                            r_date=None,
                            sort='created_date_desc',
                            **kwargs):
    session = get_session(session)
    params = {
        'sort': sort,
        'no_aggs': 'true',
        'no_highlight': 'true',
        **kwargs
    }
    if isinstance(l_date, dt.date) and isinstance(r_date, dt.date):
        params.update({
            'format': 'json',
            'date_received_max': r_date.strftime('%Y-%m-%d'),
            'date_received_min': l_date.strftime('%Y-%m-%d'),
        })

    logging.info('Requesting data from API')
    request = session.get(
        '/data-research/consumer-complaints/search/api/v1/',
        params=params,
        timeout=timeout
    )
    logging.info('Data collected with url: {}'.format(request.url))

    try:
        json_data = request.json()
    except:
        sleep_time = 10
        logging.info('error when parsing json from api')
        logging.info('will sleep for {} seconds and try again'.format(sleep_time))
        time.sleep(sleep_time)
        json_data = download_chunk_from_api(session, timeout, l_date, r_date,
                                            sort='created_date_desc', **kwargs)

    return json_data


def check_data(json_data, raise_error=False):
    for record in json_data:
        try:
            assert 'complaint-public' in record.get('_index')
            assert record.get('_type') == 'complaint'
            assert record.get('_score') == 0
            assert '_source' in record
            assert record.get('_id') == record['_source'].get('complaint_id')
        except AssertionError as err:
            logging.warning('Unexpected data format!', exc_info=True)
            logging.info(json.dumps(record))
            if raise_error:
                raise err


def store_data(dbcon,
               df_data,
               tablename,
               ignore_previous,
               l_date=None,
               r_date=None,
               **kwargs):
    with dbcon.cursor() as dbcur:
        if ignore_previous:
            type_of_data = kwargs.get('td', '')
            logging.debug('storing {} elements:\n{}\n'.format(
                type_of_data,
                df_data.head()))
            if not df_data.empty:
                psql_insert_copy(dbcon, df_data, tablename)
                return {type_of_data : df_data}
            return dict()
        else:
            existing_data = psql.read_sql('''
                SELECT *
                FROM %s
                WHERE date_received >= '%s'::date
                AND date_received < '%s'::date
            ''' % (AsIs(quote_ident(tablename, dbcur)),
                   l_date,
                   r_date),
                dbcon)

            new_data = df_data.drop('update_stamp', axis=1)
            old_data = existing_data.drop('update_stamp', axis=1)
            new_ids = set(new_data._id)
            old_ids = set(old_data._id)
            common_ids = new_ids & old_ids

            logging.info('Number of new ids, old ids, common ids:\n{}'.format(
                (len(new_ids), len(old_ids), len(common_ids))))

            updated_data = new_data[~new_data.apply(tuple, 1).isin(
                old_data.apply(tuple, 1)) & new_data['_id'].isin(common_ids)]

            updated_ids = set(updated_data._id)
            deleted_ids = old_ids - new_ids
            added_ids = new_ids - old_ids

            logging.info(
                'updated_ids (total {} elements):\n{}'.format(
                    len(updated_ids), sorted(
                        updated_ids, reverse=True)[
                        :min(
                            len(updated_ids), 50)]))
            logging.info(
                'added_ids (total {} elements):\n{}'.format(
                    len(added_ids), sorted(
                        added_ids, reverse=True)[
                        :min(
                            len(added_ids), 50)]))
            logging.info(
                'deleted_ids (total {} elements):\n{}'.format(
                    len(deleted_ids), sorted(
                        deleted_ids, reverse=True)[
                        :min(
                            len(deleted_ids), 50)]))

            already_deleted = psql.read_sql('''
                            SELECT _id
                            FROM %s
                            WHERE date_received IS null
                        ''' % (AsIs(quote_ident(tablename, dbcur))), dbcon)

            updated_rows = df_data.loc[df_data['_id'].isin(updated_ids)]
            deleted_rows = existing_data.loc[existing_data['_id'].isin(
                deleted_ids) & ~existing_data['_id'].isin(already_deleted['_id'])].copy()
            deleted_rows = deleted_rows.assign(**{col: None for col in deleted_rows if col not in
                                                  ['_id', 'complaint_id', 'update_stamp']})
            deleted_rows['update_stamp'] = dt.datetime.now()
            deleted_rows = deleted_rows.drop_duplicates(ignore_index=True)
            added_rows = df_data.loc[df_data['_id'].isin(added_ids)]

            stored_data = {}
            stored_data.update(store_data(
                dbcon,
                updated_rows,
                tablename,
                ignore_previous=True,
                td='updated'))
            stored_data.update(store_data(
                dbcon,
                deleted_rows,
                tablename,
                ignore_previous=True,
                td='deleted'))
            stored_data.update(store_data(
                dbcon,
                added_rows,
                tablename,
                ignore_previous=True,
                td='added'))
            return stored_data


def psql_insert_copy(dbcon, df_data, tablename):
    with dbcon.cursor() as dbcur:
        buffer = StringIO()
        df_data.to_csv(buffer, header=False, index=False, sep=';')
        buffer.seek(0)
        dbcur.copy_from(buffer, tablename, sep=';', null='')


def draw_graph(plot, x, y, X, title, **kwargs):
    logging.info('Drawing {} graph!'.format(title))
    if plot.get_title():
        plot.set_title(' and '.join([plot.get_title(), title]))
    else:
        plot.set_title(title)
    xmin = min(X)[0]
    xmax = max(X)[0]
    xmax += (dt.timedelta(days=1) if isinstance(xmax, dt.date) else 1)
    plot.set_ylim(0, round((max(y) + 1) * 1.1))
    plot.set_xlim(xmin, xmax)
    plot.set_xticks(list(daterange(xmin, xmax, 1
                    if (xmax - xmin) <= dt.timedelta(days=DAYS_IN_MONTH)
                    else 2 * DAYS_IN_MONTH)))
    plot.tick_params(axis='x', rotation=90)
    plot.grid()
    plot.plot(x, y, **kwargs)


def draw_counting_graph(data, path=None, graphtype=0, labels=None, separate_plots=True):
    if separate_plots:
        fig, ax = plt.subplots(2, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)
        plt1 = ax[0]
        plt2 = ax[1]
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
        plt1 = ax
        plt2 = ax

    if graphtype == 0:
        graphtype = 'countgraph'
        filename = 'count_from_{}_to_{}_graph.png'
        labels = ['updated', 'added']
    elif graphtype == 1:
        graphtype = 'companies'
        filename = 'compare_{}_and_{}_graph.png'
    else:
        return

    label1 = labels[0]
    label2 = labels[1]
    data1 = data.get(label1)
    data2 = data.get(label2)
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    if data1 is not None:
        data1['date_received'] = data1['date_received'].apply(lambda x : x.date())
        points = sorted(data1.value_counts(subset=['date_received']).to_dict().items())
        x1 = [i[0] for i in points]
        y1 = [i[1] for i in points]
    if data2 is not None:
        data2['date_received'] = data2['date_received'].apply(lambda x : x.date())
        points = sorted(data2.value_counts(subset=['date_received']).to_dict().items())
        x2 = [i[0] for i in points]
        y2 = [i[1] for i in points]

    if x := x1 + x2:
        draw_graph(plt1, x1, y1, x, '{} elements'.format(label1.upper()), color='b')
        draw_graph(plt2, x2, y2, x, '{} elements'.format(label2.upper()), color='r')
        if graphtype == 'countgraph':
            add_name = [min(x)[0].strftime('%Y-%m-%d'), max(x)[0].strftime('%Y-%m-%d')]
        elif graphtype == 'companies':
            add_name = [''.join(filter(lambda x : str.isalnum(x) or str.isspace(x), s)) for s in labels]
        path = path or os.path.dirname(os.path.realpath(__file__))
        plt.savefig(os.path.join(path, filename.format(*add_name)))


def draw_companies_graph(dbcon, tablename, company1, company2):
    with dbcon.cursor() as dbcur:
        company1_data = psql.read_sql('''
            SELECT *
            FROM %s
            WHERE company = '%s'
        ''' % (AsIs(quote_ident(tablename, dbcur)), company1), dbcon)

        company2_data = psql.read_sql('''
            SELECT *
            FROM %s
            WHERE company = '%s'
        ''' % (AsIs(quote_ident(tablename, dbcur)), company2), dbcon)

        logging.info('Number of complaints for {} company: {}'.format(company1, company1_data.shape[0]))
        logging.info('Number of complaints for {} company: {}'.format(company2, company2_data.shape[0]))

        data = {
            company1 : company1_data,
            company2 : company2_data,
        }
        draw_counting_graph(data, graphtype=1, labels=[company1, company2])


def main():
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    logging.info('Arguments: {}'.format(args))

    with connect_to_local_postgres(args.user, args.password,
                                   args.database, args.port) as dbcon:
        logging.info('PostgreSQL connected successfully')
        session = get_session()
        logging.info('Requests session created')
        if check_table_exists(dbcon, args.tablename):
            logging.info('Table was found')
            data_from_last_period = get_data_for_one_period(dbcon, session, args.tablename)
            logging.info('Data for the last month downloaded and stored')
        else:
            logging.info('Table was not found')
            first_rec = get_first_record(session)['hits']['hits'][0]['_source']
            first_rec_date = dt.datetime.strptime(
                first_rec['date_received'][:-6], '%Y-%m-%dT%H:%M:%S')
            table_values = list(first_rec.keys())
            create_table(dbcon, args.tablename, table_values)
            logging.info('Table created')
            data_from_last_period = get_data_for_all_time(dbcon, session, args.tablename, first_rec_date)
            logging.info('Data for the all time downloaded and stored')

    if args.countgraph:
        logging.info('Will now draw a graph counting updates each day')
        logging.info('Number of elements from this running:')
        for i in data_from_last_period:
            logging.info('{} elements: {}'.format(i, data_from_last_period[i].shape[0]))
        draw_counting_graph(data_from_last_period)

    with connect_to_local_postgres(args.user, args.password,
                                   args.database, args.port) as dbcon:
        if args.companies:
            logging.info('Will now draw a graph counting complaints for {} companies'.format(
                ' and '.join(args.companies)
            ))
            draw_companies_graph(dbcon, args.tablename, *args.companies)


main()
