#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import requests


class ISICApi(object):
    def __init__(self, hostname='https://isic-archive.com',
                 username=None, password=None):
        self.baseUrl = '%s/api/v1' % hostname
        self.authToken = None

        if username is not None:
            if password is None:
                password = input('Password for user "%s":' % username)
            self.authToken = self._login(username, password)

    def _makeUrl(self, endpoint):
        return '%s/%s' % (self.baseUrl, endpoint)

    def _login(self, username, password):
        authResponse = requests.get(
            self._makeUrl('user/authentication'),
            auth=(username, password)
        )
        if not authResponse.ok:
            raise Exception('Login error: %s' % authResponse.json()['message'])

        authToken = authResponse.json()['authToken']['token']
        return authToken

    def get(self, endpoint):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.get(url, headers=headers)

    def getJson(self, endpoint):
        return self.get(endpoint).json()

    def getJsonList(self, endpoint):
        endpoint += '&' if '?' in endpoint else '?'
        LIMIT = 50
        offset = 0
        while True:
            resp = self.get(
                '%slimit=%d&offset=%d' % (endpoint, LIMIT, offset)
            ).json()
            if not resp:
                break
            for elem in resp:
                yield elem
            offset += LIMIT
